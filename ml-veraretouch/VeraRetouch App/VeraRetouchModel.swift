//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import CoreImage
import CoreML
import Darwin
import VeraRetouchCore
import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import MLXVLM
import Tokenizers

private struct RetouchModuleTimings {
    var inputPrepareMs: Double
    var llmGenerateMs: Double
    var hiddenStateMs: Double
    var modelLoadMs: Double
    var retouchHeadMs: Double
    var decoderInputPrepMs: Double
    var retouchDecoderMs: Double
    var imagePostprocessMs: Double

    var totalMs: Double {
        inputPrepareMs
            + llmGenerateMs
            + hiddenStateMs
            + modelLoadMs
            + retouchHeadMs
            + decoderInputPrepMs
            + retouchDecoderMs
            + imagePostprocessMs
    }
}

private func nowNs() -> UInt64 {
    DispatchTime.now().uptimeNanoseconds
}

private func elapsedMs(since startNs: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - startNs) / 1_000_000.0
}

private func formatMs(_ value: Double) -> String {
    "\(Int(value.rounded())) ms"
}

@Observable
@MainActor
class VeraRetouchModel {

    public var running = false
    public var modelInfo = ""
    public var output = ""
    public var isStreamingOutput = false
    public var promptTime: String = ""
    public var retouchPipelineTime: String = ""
    public var retouchTimingBreakdown: [String] = []
    public var sourceImage: CGImage?
    public var retouchedImage: CGImage?
    public var lastLUTCubeData: Data?
    public var lastLUTDimension: Int?
    public var hasCachedRetouchState: Bool { cachedRetouchState != nil }

    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    private let modelConfiguration = VeraRetouch.modelConfiguration

    /// default sampling parameters
    let defaultTemperature: Float = 0.2
    let defaultTopP: Float = 0.8
    let maxTokens = 2048

    /// update the display every N tokens -- 4 looks like it updates continuously
    /// and is low overhead.  observed ~15% reduction in tokens/s when updating
    /// on every token
    let displayEveryNTokens = 4

    private var loadState = LoadState.idle
    private var currentTask: Task<Void, Never>?
    private static let previewContext = CIContext()
    private var cachedRetouchState: SamanthaRetouchEngine.CachedRetouchState?
    private var runTokenCounter: UInt64 = 0
    private var activeRunToken: UInt64 = 0

    enum EvaluationState: String, CaseIterable {
        case idle = "Idle"
        case processingPrompt = "Processing Prompt"
        case generatingResponse = "Generating Response"
    }

    public var evaluationState = EvaluationState.idle
    
    private var isUnsupportedSimulatorRuntime: Bool {
        #if os(iOS) && targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
    }

    public init() {
        VeraRetouch.register(modelFactory: VLMModelFactory.shared)
    }

    private func beginRunToken() -> UInt64 {
        runTokenCounter &+= 1
        activeRunToken = runTokenCounter
        return activeRunToken
    }

    private func invalidateRunToken() {
        runTokenCounter &+= 1
        activeRunToken = runTokenCounter
    }

    private func isActiveRunToken(_ token: UInt64) -> Bool {
        activeRunToken == token
    }

    private func _load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            // limit the buffer cache
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let modelContainer = try await VLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) {
                [modelConfiguration] progress in
                Task { @MainActor in
                    self.modelInfo =
                        "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            self.modelInfo = "Loaded"
            loadState = .loaded(modelContainer)
            return modelContainer

        case .loaded(let modelContainer):
            return modelContainer
        }
    }

    public func load() async {
        if isUnsupportedSimulatorRuntime {
            self.modelInfo = "iOS Simulator is not supported for Samantha inference. Please run on a real iPhone/iPad device."
            return
        }
        do {
            _ = try await _load()
        } catch {
            self.modelInfo = "Error loading model: \(error)"
        }
    }

    public func generate(
        _ userInput: UserInput,
        retouchMask: [Float32]? = nil,
        lutDimension: Int? = nil,
        temperature: Float? = nil,
        topP: Float? = nil
    ) async -> Task<Void, Never> {
        if isUnsupportedSimulatorRuntime {
            self.running = false
            self.evaluationState = .idle
            self.output =
                "Current runtime is iOS Simulator, which is unsupported for this MLX/CoreML pipeline. Please run on a physical iPhone/iPad."
            return Task {}
        }

        if let currentTask, running {
            return currentTask
        }

        let runToken = beginRunToken()
        running = true
        isStreamingOutput = true
        // Cancel any existing task
        currentTask?.cancel()

        // Create new task and store reference
        let task = Task {
            do {
                let modelContainer = try await _load()

                // each time you generate you will get something new
                MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))
                
                // Check if task was cancelled
                if Task.isCancelled { return }

                let result = try await modelContainer.perform { context in
                    // Measure the time it takes to prepare the input
                    
                    Task { @MainActor in
                        guard self.isActiveRunToken(runToken) else { return }
                        evaluationState = .processingPrompt
                    }

                    let llmStart = Date()
                    let inputPrepareStart = nowNs()
                    let input = try await context.processor.prepare(input: userInput)
                    let inputPrepareMs = elapsedMs(since: inputPrepareStart)
                    var generatedTokens = [Int]()
                    
                    var seenFirstToken = false

                    // VeraRetouch generates the output
                    let llmGenerateStart = nowNs()
                    let samplingParameters = GenerateParameters(
                        temperature: temperature ?? defaultTemperature,
                        topP: topP ?? defaultTopP
                    )
                    let result = try MLXLMCommon.generate(
                        input: input, parameters: samplingParameters, context: context
                    ) { tokens in
                        generatedTokens = tokens
                        // Check if task was cancelled
                        if Task.isCancelled {
                            return .stop
                        }

                        if !seenFirstToken {
                            seenFirstToken = true
                            
                            // produced first token, update the time to first token,
                            // the processing state and start displaying the text
                            let llmDuration = Date().timeIntervalSince(llmStart)
                            let text = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                guard self.isActiveRunToken(runToken) else { return }
                                evaluationState = .generatingResponse
                                self.output = text
                                self.promptTime = "\(Int(llmDuration * 1000)) ms"
                            }
                        }

                        // Show the text in the view as it generates
                        if tokens.count % displayEveryNTokens == 0 {
                            let text = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                guard self.isActiveRunToken(runToken) else { return }
                                self.output = text
                            }
                        }

                        if tokens.count >= maxTokens {
                            return .stop
                        } else {
                            return .more
                        }
                    }
                    let llmGenerateMs = elapsedMs(since: llmGenerateStart)
                    Task { @MainActor in
                        guard self.isActiveRunToken(runToken) else { return }
                        self.isStreamingOutput = false
                    }

                    var sourcePreview: CGImage?
                    var retouchedImage: CGImage?
                    var timings: RetouchModuleTimings?
                    var lutCubeData: Data?
                    var lutDim: Int?
                    var cachedState: SamanthaRetouchEngine.CachedRetouchState?
                    if let sourceImage = try? userInput.images.first?.asCIImage(),
                        let model = context.model as? VeraRetouch
                    {
                        sourcePreview = Self.previewContext.createCGImage(
                            sourceImage, from: sourceImage.extent)
                        let retouchResult = try? SamanthaRetouchEngine.generateRetouchedImage(
                            model: model,
                            input: input,
                            tokenizer: context.tokenizer,
                            generatedTokens: generatedTokens,
                            sourceImage: sourceImage,
                            retouchMask: retouchMask,
                            lutDimension: lutDimension
                        )
                        retouchedImage = retouchResult?.image
                        lutCubeData = retouchResult?.lutCubeData
                        lutDim = retouchResult?.lutDimension
                        cachedState = retouchResult?.cachedState
                        if let retouchTimings = retouchResult?.timings {
                            timings = RetouchModuleTimings(
                                inputPrepareMs: inputPrepareMs,
                                llmGenerateMs: llmGenerateMs,
                                hiddenStateMs: retouchTimings.hiddenStateMs,
                                modelLoadMs: retouchTimings.modelLoadMs,
                                retouchHeadMs: retouchTimings.retouchHeadMs,
                                decoderInputPrepMs: retouchTimings.decoderInputPrepMs,
                                retouchDecoderMs: retouchTimings.retouchDecoderMs,
                                imagePostprocessMs: retouchTimings.imagePostprocessMs
                            )
                        }
                    }

                    return (result, retouchedImage, sourcePreview, timings, lutCubeData, lutDim, cachedState)
                }
                
                // Check if task was cancelled before updating UI
                if !Task.isCancelled {
                    guard self.isActiveRunToken(runToken) else { return }
                    self.output = Self.reviewOutputText(result.0.output, with: retouchMask)
                    self.retouchedImage = result.1
                    self.sourceImage = result.2
                    self.lastLUTCubeData = result.4
                    self.lastLUTDimension = result.5
                    self.cachedRetouchState = result.6
                    if let timings = result.3 {
                        self.retouchPipelineTime = formatMs(timings.totalMs)
                        self.retouchTimingBreakdown = [
                            "Total: \(formatMs(timings.totalMs))",
                            "Input prep: \(formatMs(timings.inputPrepareMs))",
                            "LLM generation: \(formatMs(timings.llmGenerateMs))",
                            "Hidden states: \(formatMs(timings.hiddenStateMs))",
                            "Model load: \(formatMs(timings.modelLoadMs))",
                            "Retouch head: \(formatMs(timings.retouchHeadMs))",
                            "Decoder input prep: \(formatMs(timings.decoderInputPrepMs))",
                            "Retouch decoder: \(formatMs(timings.retouchDecoderMs))",
                            "Image postprocess: \(formatMs(timings.imagePostprocessMs))",
                        ]
                    } else {
                        self.retouchPipelineTime = ""
                        self.retouchTimingBreakdown = []
                    }
                }

            } catch {
                if !Task.isCancelled {
                    guard self.isActiveRunToken(runToken) else { return }
                    output = "Failed: \(error)"
                    sourceImage = nil
                    retouchedImage = nil
                    retouchPipelineTime = ""
                    retouchTimingBreakdown = []
                    lastLUTCubeData = nil
                    lastLUTDimension = nil
                    cachedRetouchState = nil
                }
            }

            guard self.isActiveRunToken(runToken) else { return }
            if evaluationState == .generatingResponse {
                evaluationState = .idle
            }
            isStreamingOutput = false
            running = false
        }
        
        currentTask = task
        return task
    }
    
    private static func reviewOutputText(_ text: String, with mask: [Float32]?) -> String {
        guard let mask, mask.count >= 3 else { return text }
        if mask[0] >= 0.5, mask[1] >= 0.5, mask[2] >= 0.5 { return text }

        var reviewed = text

        // Professional plan tags (from model output format):
        // <plan_light_start> ... <plan_light_end>
        // <plan_globalcolor_start> ... <plan_globalcolor_end>
        // <plan_specificcolor_start> ... <plan_specificcolor_end>
        let groupTagPairs: [(Float32, String, String)] = [
            (mask[0], "<plan_light_start>", "<plan_light_end>"),
            (mask[1], "<plan_globalcolor_start>", "<plan_globalcolor_end>"),
            (mask[2], "<plan_specificcolor_start>", "<plan_specificcolor_end>"),
        ]
        for (m, startTag, endTag) in groupTagPairs where m < 0.5 {
            reviewed = clearTaggedBlockContent(in: reviewed, startTag: startTag, endTag: endTag)
        }
        return reviewed
    }

    private static func clearTaggedBlockContent(in text: String, startTag: String, endTag: String) -> String {
        let escapedStart = NSRegularExpression.escapedPattern(for: startTag)
        let escapedEnd = NSRegularExpression.escapedPattern(for: endTag)
        let pattern = "(\(escapedStart))[\\s\\S]*?(\(escapedEnd))"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return text
        }
        let range = NSRange(text.startIndex..., in: text)
        // Keep tags, clear only the content in-between.
        return regex.stringByReplacingMatches(
            in: text,
            options: [],
            range: range,
            withTemplate: "$1$2"
        )
    }

    public func cancel() {
        invalidateRunToken()
        currentTask?.cancel()
        currentTask = nil
        running = false
        isStreamingOutput = false
        output = ""
        promptTime = ""
        retouchPipelineTime = ""
        retouchTimingBreakdown = []
        sourceImage = nil
        retouchedImage = nil
        lastLUTCubeData = nil
        lastLUTDimension = nil
        cachedRetouchState = nil
    }

    public func stopGeneration() {
        invalidateRunToken()
        currentTask?.cancel()
        currentTask = nil
        isStreamingOutput = false
        if evaluationState != .idle {
            evaluationState = .idle
        }
        running = false
    }

    public func generateAndWait(
        _ userInput: UserInput,
        retouchMask: [Float32]? = nil,
        lutDimension: Int? = nil,
        temperature: Float? = nil,
        topP: Float? = nil
    ) async {
        let task = await generate(
            userInput,
            retouchMask: retouchMask,
            lutDimension: lutDimension,
            temperature: temperature,
            topP: topP
        )
        _ = await task.result
    }

    public func rerenderFromCachedRetouch(
        retouchMask: [Float32]? = nil,
        lutDimension: Int? = nil
    ) async -> Bool {
        guard let cachedRetouchState else { return false }
        if running { return false }
        _ = beginRunToken()
        running = true
        isStreamingOutput = false
        defer { running = false }

        do {
            let result = try SamanthaRetouchEngine.rerenderRetouchedImage(
                cachedState: cachedRetouchState,
                retouchMask: retouchMask,
                lutDimension: lutDimension
            )
            self.retouchedImage = result.image
            self.sourceImage = Self.previewContext.createCGImage(
                cachedRetouchState.sourceImage,
                from: cachedRetouchState.sourceImage.extent
            )
            self.lastLUTCubeData = result.lutCubeData
            self.lastLUTDimension = result.lutDimension
            self.retouchPipelineTime = formatMs(
                result.timings.modelLoadMs
                    + result.timings.decoderInputPrepMs
                    + result.timings.retouchDecoderMs
                    + result.timings.imagePostprocessMs
            )
            self.retouchTimingBreakdown = [
                "Rerender (decoder only)",
                "Model load: \(formatMs(result.timings.modelLoadMs))",
                "Decoder input prep: \(formatMs(result.timings.decoderInputPrepMs))",
                "Retouch decoder: \(formatMs(result.timings.retouchDecoderMs))",
                "Image postprocess: \(formatMs(result.timings.imagePostprocessMs))",
            ]
            return result.image != nil
        } catch {
            self.output = "Rerender failed: \(error)"
            return false
        }
    }

    public func referenceRetouch(
        sourceImage: CIImage,
        referenceBeforeImage: CIImage,
        referenceAfterImage: CIImage,
        lutDimension: Int? = nil
    ) async -> Bool {
        if running { return false }
        _ = beginRunToken()
        running = true
        isStreamingOutput = false
        output = ""
        retouchedImage = nil
        retouchPipelineTime = ""
        retouchTimingBreakdown = []
        lastLUTCubeData = nil
        lastLUTDimension = nil
        cachedRetouchState = nil
        defer { running = false }

        do {
            let result = try SamanthaRetouchEngine.generateReferenceRetouchedImage(
                sourceImage: sourceImage,
                referenceBeforeImage: referenceBeforeImage,
                referenceAfterImage: referenceAfterImage,
                lutDimension: lutDimension
            )
            self.sourceImage = Self.previewContext.createCGImage(sourceImage, from: sourceImage.extent)
            self.retouchedImage = result.image
            self.lastLUTCubeData = result.lutCubeData
            self.lastLUTDimension = result.lutDimension
            self.cachedRetouchState = result.cachedState
            self.output = "Reference retouch completed."
            self.retouchPipelineTime = formatMs(
                result.timings.modelLoadMs
                    + result.timings.retouchHeadMs
                    + result.timings.decoderInputPrepMs
                    + result.timings.retouchDecoderMs
                    + result.timings.imagePostprocessMs
            )
            self.retouchTimingBreakdown = [
                "Reference retouch",
                "Model load: \(formatMs(result.timings.modelLoadMs))",
                "Ref encoder: \(formatMs(result.timings.retouchHeadMs))",
                "Decoder input prep: \(formatMs(result.timings.decoderInputPrepMs))",
                "Retouch decoder: \(formatMs(result.timings.retouchDecoderMs))",
                "Image postprocess: \(formatMs(result.timings.imagePostprocessMs))",
            ]
            return result.image != nil
        } catch {
            self.output = "Reference retouch failed: \(error)"
            return false
        }
    }
}

private enum SamanthaRetouchEngine {
    static let lightToken = "<retouch_light>"
    static let colorTempToken = "<retouch_color&temp>"
    static let colorMixerToken = "<retouch_colormixer>"
    static let fallbackLightTokenId = 151646
    static let fallbackColorTempTokenId = 151647
    static let fallbackColorMixerTokenId = 151648

    struct CachedRetouchState {
        let sourceImage: CIImage
        let baseRetouchLatent: MLMultiArray
    }

    static func generateRetouchedImage(
        model: VeraRetouch,
        input: LMInput,
        tokenizer: any Tokenizer,
        generatedTokens: [Int],
        sourceImage: CIImage,
        retouchMask: [Float32]? = nil,
        lutDimension: Int? = nil
    ) throws -> (
        image: CGImage?, timings: RetouchTimings, lutCubeData: Data?, lutDimension: Int?,
        cachedState: CachedRetouchState?
    ) {
        guard let tokenIds = resolveRetouchTokenIds(tokenizer: tokenizer) else {
            return (
                nil,
                RetouchTimings(
                    hiddenStateMs: 0,
                    modelLoadMs: 0,
                    retouchHeadMs: 0,
                    decoderInputPrepMs: 0,
                    retouchDecoderMs: 0,
                    imagePostprocessMs: 0
                ),
                nil,
                nil,
                nil
            )
        }
        let (lightTokenId, colorTempTokenId, colorMixerTokenId) = tokenIds

        let hiddenStateStart = nowNs()
        let (fullTokens, hiddenStates) = model.finalHiddenStates(
            input: input, generatedTokens: generatedTokens)
        guard let lightIndex = fullTokens.lastIndex(of: lightTokenId),
            let colorTempIndex = fullTokens.lastIndex(of: colorTempTokenId),
            let colorMixerIndex = fullTokens.lastIndex(of: colorMixerTokenId),
            lightIndex > 0,
            colorTempIndex > 0,
            colorMixerIndex > 0
        else {
            return (
                nil,
                RetouchTimings(
                    hiddenStateMs: elapsedMs(since: hiddenStateStart),
                    modelLoadMs: 0,
                    retouchHeadMs: 0,
                    decoderInputPrepMs: 0,
                    retouchDecoderMs: 0,
                    imagePostprocessMs: 0
                ),
                nil,
                nil,
                nil
            )
        }
        let hiddenStateMs = elapsedMs(since: hiddenStateStart)

        // Align with Python collector semantics: use hidden state from the
        // position immediately before each emitted retouch token.
        let lightState = hiddenStates[0..., lightIndex - 1, 0...]
        let colorTempState = hiddenStates[0..., colorTempIndex - 1, 0...]
        let colorMixerState = hiddenStates[0..., colorMixerIndex - 1, 0...]
        // Keep shape [1, hidden*3] to match retouch_head input spec.
        let retouchInput = concatenated([lightState, colorTempState, colorMixerState], axis: -1)

        let modelLoadStart = nowNs()
        let retouchHeadModel = try loadModel(named: "retouch_head")
        let retouchDecoderModel = try loadModel(named: "retouch_decoder")
        let modelLoadMs = elapsedMs(since: modelLoadStart)

        let retouchHeadStart = nowNs()
        let headInput = try mlMultiArray(from: retouchInput)
        let headProvider = try MLDictionaryFeatureProvider(
            dictionary: ["input": MLFeatureValue(multiArray: headInput)])
        let headOutput = try retouchHeadModel.prediction(from: headProvider)
        guard let retouchLatent = headOutput.featureValue(for: "output")?.multiArrayValue else {
            return (
                nil,
                RetouchTimings(
                    hiddenStateMs: hiddenStateMs,
                    modelLoadMs: modelLoadMs,
                    retouchHeadMs: elapsedMs(since: retouchHeadStart),
                    decoderInputPrepMs: 0,
                    retouchDecoderMs: 0,
                    imagePostprocessMs: 0
                ),
                nil,
                nil,
                nil
            )
        }
        let retouchHeadMs = elapsedMs(since: retouchHeadStart)

        let baseRetouchLatent = try cloneMultiArray(retouchLatent)

        let renderResult = try renderImageFromLatent(
            sourceImage: sourceImage,
            baseRetouchLatent: baseRetouchLatent,
            retouchMask: retouchMask,
            lutDimension: lutDimension,
            retouchDecoderModel: retouchDecoderModel
        )

        guard let image = renderResult.image else {
            return (
                nil,
                RetouchTimings(
                    hiddenStateMs: hiddenStateMs,
                    modelLoadMs: modelLoadMs,
                    retouchHeadMs: retouchHeadMs,
                    decoderInputPrepMs: renderResult.decoderInputPrepMs,
                    retouchDecoderMs: renderResult.retouchDecoderMs,
                    imagePostprocessMs: renderResult.imagePostprocessMs
                ),
                renderResult.lutCubeData,
                renderResult.lutDimension,
                CachedRetouchState(sourceImage: sourceImage, baseRetouchLatent: baseRetouchLatent)
            )
        }

        return (
            image,
            RetouchTimings(
                hiddenStateMs: hiddenStateMs,
                modelLoadMs: modelLoadMs,
                retouchHeadMs: retouchHeadMs,
                decoderInputPrepMs: renderResult.decoderInputPrepMs,
                retouchDecoderMs: renderResult.retouchDecoderMs,
                imagePostprocessMs: renderResult.imagePostprocessMs
            ),
            renderResult.lutCubeData,
            renderResult.lutDimension,
            CachedRetouchState(sourceImage: sourceImage, baseRetouchLatent: baseRetouchLatent)
        )
    }

    static func rerenderRetouchedImage(
        cachedState: CachedRetouchState,
        retouchMask: [Float32]? = nil,
        lutDimension: Int? = nil
    ) throws -> (image: CGImage?, timings: RetouchTimings, lutCubeData: Data?, lutDimension: Int?) {
        let modelLoadStart = nowNs()
        let retouchDecoderModel = try loadModel(named: "retouch_decoder")
        let modelLoadMs = elapsedMs(since: modelLoadStart)

        let renderResult = try renderImageFromLatent(
            sourceImage: cachedState.sourceImage,
            baseRetouchLatent: cachedState.baseRetouchLatent,
            retouchMask: retouchMask,
            lutDimension: lutDimension,
            retouchDecoderModel: retouchDecoderModel
        )

        return (
            renderResult.image,
            RetouchTimings(
                hiddenStateMs: 0,
                modelLoadMs: modelLoadMs,
                retouchHeadMs: 0,
                decoderInputPrepMs: renderResult.decoderInputPrepMs,
                retouchDecoderMs: renderResult.retouchDecoderMs,
                imagePostprocessMs: renderResult.imagePostprocessMs
            ),
            renderResult.lutCubeData,
            renderResult.lutDimension
        )
    }

    static func generateReferenceRetouchedImage(
        sourceImage: CIImage,
        referenceBeforeImage: CIImage,
        referenceAfterImage: CIImage,
        lutDimension: Int? = nil
    ) throws -> (
        image: CGImage?, timings: RetouchTimings, lutCubeData: Data?, lutDimension: Int?,
        cachedState: CachedRetouchState?
    ) {
        let modelLoadStart = nowNs()
        let refEncoderModel = try loadModel(named: "ref_retouch_encoder")
        let refDecoderModel = try loadModel(named: "ref_retouch_decoder")
        let modelLoadMs = elapsedMs(since: modelLoadStart)

        let refEncoderStart = nowNs()
        let refBeforeData = prepareImageInput(referenceBeforeImage, width: 512, height: 512)
        let refAfterData = prepareImageInput(referenceAfterImage, width: 512, height: 512)
        let refBeforeInput = try mlMultiArrayFromNCHW(
            data: refBeforeData.data,
            height: refBeforeData.height,
            width: refBeforeData.width
        )
        let refAfterInput = try mlMultiArrayFromNCHW(
            data: refAfterData.data,
            height: refAfterData.height,
            width: refAfterData.width
        )
        let mask = try MLMultiArray(shape: [1, 3].map { NSNumber(value: $0) }, dataType: .float32)
        let maskPtr = mask.dataPointer.bindMemory(to: Float32.self, capacity: 3)
        maskPtr[0] = 1.0
        maskPtr[1] = 1.0
        maskPtr[2] = 1.0

        let encoderProvider = try MLDictionaryFeatureProvider(
            dictionary: [
                "ref_before_img": MLFeatureValue(multiArray: refBeforeInput),
                "ref_after_img": MLFeatureValue(multiArray: refAfterInput),
                "mask": MLFeatureValue(multiArray: mask),
            ])
        let encoderOutput = try refEncoderModel.prediction(from: encoderProvider)
        guard let retouchLatent = encoderOutput.featureValue(for: "retouch_latent")?.multiArrayValue else {
            return (
                nil,
                RetouchTimings(
                    hiddenStateMs: 0,
                    modelLoadMs: modelLoadMs,
                    retouchHeadMs: elapsedMs(since: refEncoderStart),
                    decoderInputPrepMs: 0,
                    retouchDecoderMs: 0,
                    imagePostprocessMs: 0
                ),
                nil,
                nil,
                nil
            )
        }
        let refEncoderMs = elapsedMs(since: refEncoderStart)
        let baseRetouchLatent = try cloneMultiArray(retouchLatent)

        let renderResult = try renderImageFromLatent(
            sourceImage: sourceImage,
            baseRetouchLatent: baseRetouchLatent,
            retouchMask: nil,
            lutDimension: lutDimension,
            retouchDecoderModel: refDecoderModel
        )

        return (
            renderResult.image,
            RetouchTimings(
                hiddenStateMs: 0,
                modelLoadMs: modelLoadMs,
                retouchHeadMs: refEncoderMs,
                decoderInputPrepMs: renderResult.decoderInputPrepMs,
                retouchDecoderMs: renderResult.retouchDecoderMs,
                imagePostprocessMs: renderResult.imagePostprocessMs
            ),
            renderResult.lutCubeData,
            renderResult.lutDimension,
            CachedRetouchState(sourceImage: sourceImage, baseRetouchLatent: baseRetouchLatent)
        )
    }

    struct RetouchTimings {
        var hiddenStateMs: Double
        var modelLoadMs: Double
        var retouchHeadMs: Double
        var decoderInputPrepMs: Double
        var retouchDecoderMs: Double
        var imagePostprocessMs: Double
    }

    static func cloneMultiArray(_ input: MLMultiArray) throws -> MLMultiArray {
        guard input.dataType == .float32 else {
            throw NSError(
                domain: "SamanthaRetouchEngine",
                code: 12,
                userInfo: [NSLocalizedDescriptionKey: "Unsupported latent datatype \(input.dataType)"]
            )
        }
        let output = try MLMultiArray(shape: input.shape, dataType: .float32)
        let count = input.count
        let src = input.dataPointer.bindMemory(to: Float32.self, capacity: count)
        let dst = output.dataPointer.bindMemory(to: Float32.self, capacity: count)
        for i in 0..<count {
            dst[i] = src[i]
        }
        return output
    }

    static func renderImageFromLatent(
        sourceImage: CIImage,
        baseRetouchLatent: MLMultiArray,
        retouchMask: [Float32]?,
        lutDimension: Int?,
        retouchDecoderModel: MLModel
    ) throws -> (
        image: CGImage?, decoderInputPrepMs: Double, retouchDecoderMs: Double,
        imagePostprocessMs: Double, lutCubeData: Data?, lutDimension: Int?
    ) {
        let retouchLatent = try cloneMultiArray(baseRetouchLatent)
        if let mask = retouchMask, mask.count >= 3 {
            applyRetouchMask(retouchLatent, mask: [mask[0], mask[1], mask[2]])
        }

        let decoderInputPrepStart = nowNs()
        let decoderInputPrepMs: Double
        var image: CGImage?
        var retouchDecoderMs: Double
        var imagePostprocessMs: Double
        var outputLUTCubeData: Data?
        var outputLUTDimension: Int?

        if let lutDimension, lutDimension > 1 {
            let lutAtlas = buildIdentityLUTAtlasNCHW(dimension: lutDimension)
            decoderInputPrepMs = elapsedMs(since: decoderInputPrepStart)

            let retouchDecoderStart = nowNs()
            guard
                let outputAtlas = try runRetouchDecoderTiled(
                    model: retouchDecoderModel,
                    inputData: lutAtlas.data,
                    inputHeight: lutAtlas.height,
                    inputWidth: lutAtlas.width,
                    retouchLatent: retouchLatent,
                    tileSize: 512
                )
            else {
                return (nil, decoderInputPrepMs, elapsedMs(since: retouchDecoderStart), 0, nil, nil)
            }
            retouchDecoderMs = elapsedMs(since: retouchDecoderStart)

            let imagePostprocessStart = nowNs()
            outputLUTCubeData = build3DLUTCubeDataFromAtlas(atlasNCHW: outputAtlas, dimension: lutDimension)
            outputLUTDimension = lutDimension
            image = apply3DLUTFromAtlas(
                atlasNCHW: outputAtlas,
                dimension: lutDimension,
                sourceImage: sourceImage
            )
            imagePostprocessMs = elapsedMs(since: imagePostprocessStart)

            if image == nil, lutDimension > 33 {
                let lut33DecoderStart = nowNs()
                let lut33Atlas = buildIdentityLUTAtlasNCHW(dimension: 33)
                if let outputAtlas33 = try runRetouchDecoderTiled(
                    model: retouchDecoderModel,
                    inputData: lut33Atlas.data,
                    inputHeight: lut33Atlas.height,
                    inputWidth: lut33Atlas.width,
                    retouchLatent: retouchLatent,
                    tileSize: 512
                ) {
                    retouchDecoderMs += elapsedMs(since: lut33DecoderStart)
                    let lut33PostStart = nowNs()
                    outputLUTCubeData = build3DLUTCubeDataFromAtlas(atlasNCHW: outputAtlas33, dimension: 33)
                    outputLUTDimension = 33
                    image = apply3DLUTFromAtlas(
                        atlasNCHW: outputAtlas33,
                        dimension: 33,
                        sourceImage: sourceImage
                    )
                    imagePostprocessMs += elapsedMs(since: lut33PostStart)
                } else {
                    retouchDecoderMs += elapsedMs(since: lut33DecoderStart)
                }
            }

            if image == nil {
                let fallbackStart = nowNs()
                image = try runRetouchDecoderTiledFromCIImage(
                    model: retouchDecoderModel,
                    sourceImage: sourceImage,
                    retouchLatent: retouchLatent,
                    tileSize: 512
                )
                imagePostprocessMs += elapsedMs(since: fallbackStart)
            }
        } else {
            decoderInputPrepMs = elapsedMs(since: decoderInputPrepStart)
            let retouchDecoderStart = nowNs()
            image = try runRetouchDecoderTiledFromCIImage(
                model: retouchDecoderModel,
                sourceImage: sourceImage,
                retouchLatent: retouchLatent,
                tileSize: 512
            )
            retouchDecoderMs = elapsedMs(since: retouchDecoderStart)
            let imagePostprocessStart = nowNs()
            imagePostprocessMs = elapsedMs(since: imagePostprocessStart)
        }

        return (
            image,
            decoderInputPrepMs,
            retouchDecoderMs,
            imagePostprocessMs,
            outputLUTCubeData,
            outputLUTDimension
        )
    }

    static func resolveRetouchTokenIds(tokenizer: any Tokenizer) -> (Int, Int, Int)? {
        // Primary path: tokenizer lookup, equivalent to Python's
        // encode(..., add_special_tokens=False) for special-token strings.
        if let lightId = tokenizer.encode(text: lightToken).first,
            let colorTempId = tokenizer.encode(text: colorTempToken).first,
            let colorMixerId = tokenizer.encode(text: colorMixerToken).first
        {
            return (lightId, colorTempId, colorMixerId)
        }

        // Secondary path: read tokenizer added tokens from bundled model files.
        if let fromAddedTokens = retouchTokenIdsFromAddedTokens() {
            return fromAddedTokens
        }

        // Final fallback: Samantha checkpoints in this repo use these IDs.
        return (fallbackLightTokenId, fallbackColorTempTokenId, fallbackColorMixerTokenId)
    }

    static func retouchTokenIdsFromAddedTokens() -> (Int, Int, Int)? {
        let bundle = Bundle(for: VeraRetouch.self)
        let candidates: [URL?] = [
            bundle.url(forResource: "added_tokens", withExtension: "json"),
            bundle.url(forResource: "added_tokens", withExtension: "json", subdirectory: "model"),
            bundle.url(forResource: "config", withExtension: "json")?
                .deletingLastPathComponent()
                .appendingPathComponent("added_tokens.json"),
        ]
        guard let url = candidates.compactMap({ $0 }).first,
            let data = try? Data(contentsOf: url),
            let map = try? JSONDecoder().decode([String: Int].self, from: data),
            let light = map[lightToken],
            let colorTemp = map[colorTempToken],
            let colorMixer = map[colorMixerToken]
        else {
            return nil
        }
        return (light, colorTemp, colorMixer)
    }
    
    static func applyRetouchMask(_ latent: MLMultiArray, mask: [Float32]) {
        let count = latent.count
        guard count > 0 else { return }
        let perChannel = count / 3
        guard perChannel > 0, perChannel * 3 == count else { return }
        let ptr = latent.dataPointer.bindMemory(to: Float32.self, capacity: count)
        for c in 0 ..< 3 {
            let m = mask[c]
            let base = c * perChannel
            for i in 0 ..< perChannel {
                ptr[base + i] *= m
            }
        }
    }

    static func prepareDecoderInput(from image: CIImage) -> (data: [Float32], height: Int, width: Int) {
        let extent = image.extent
        let w = Int(extent.width.rounded())
        let h = Int(extent.height.rounded())
        return prepareImageInput(image, width: w, height: h)
    }

    static func prepareImageInput(_ image: CIImage, width: Int, height: Int) -> (
        data: [Float32], height: Int, width: Int
    ) {
        let w = width
        let h = height
        let sourceExtent = image.extent.integral
        let normalized = image
            .transformed(by: CGAffineTransform(translationX: -sourceExtent.minX, y: -sourceExtent.minY))
            .transformed(
                by: CGAffineTransform(
                    scaleX: CGFloat(w) / max(sourceExtent.width, 1),
                    y: CGFloat(h) / max(sourceExtent.height, 1)
                )
            )
        let bytesPerPixel = 4
        let bytesPerRow = w * bytesPerPixel

        var rgba = Data(count: h * bytesPerRow)
        let renderColorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        let renderContext = CIContext(options: nil)

        rgba.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            renderContext.render(
                normalized,
                toBitmap: baseAddress,
                rowBytes: bytesPerRow,
                bounds: CGRect(x: 0, y: 0, width: w, height: h),
                format: .RGBA8,
                colorSpace: renderColorSpace
            )
        }

        var data = [Float32](repeating: 0, count: 3 * h * w)
        rgba.withUnsafeBytes { ptr in
            guard let pixels = ptr.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for y in 0 ..< h {
                for x in 0 ..< w {
                    let src = y * bytesPerRow + x * bytesPerPixel
                    let r = Float32(pixels[src + 0]) / 255.0
                    let g = Float32(pixels[src + 1]) / 255.0
                    let b = Float32(pixels[src + 2]) / 255.0

                    let rIdx = ((0 * h + y) * w) + x
                    let gIdx = ((1 * h + y) * w) + x
                    let bIdx = ((2 * h + y) * w) + x
                    data[rIdx] = r * 2.0 - 1.0
                    data[gIdx] = g * 2.0 - 1.0
                    data[bIdx] = b * 2.0 - 1.0
                }
            }
        }

        return (data, h, w)
    }

    static func loadModel(named name: String) throws -> MLModel {
        let bundle = Bundle(for: VeraRetouch.self)
        let candidates: [URL?] = [
            bundle.url(forResource: name, withExtension: "mlmodelc"),
            bundle.url(forResource: name, withExtension: "mlpackage"),
            bundle.url(forResource: name, withExtension: "mlmodelc", subdirectory: "model"),
            bundle.url(forResource: name, withExtension: "mlpackage", subdirectory: "model"),
            bundle.url(forResource: "config", withExtension: "json")?
                .deletingLastPathComponent()
                .appendingPathComponent("\(name).mlpackage"),
        ]
        guard let url = candidates.compactMap({ $0 }).first else {
            throw NSError(
                domain: "SamanthaRetouchEngine", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Missing \(name) CoreML model in bundle"]
            )
        }
        return try MLModel(contentsOf: url)
    }

    static func mlMultiArray(from array: MLXArray) throws -> MLMultiArray {
        let floatArray = array.asType(.float32)
        let flat = floatArray.asArray(Float32.self)
        let shape = (0 ..< floatArray.ndim).map { NSNumber(value: floatArray.dim($0)) }
        let out = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = out.dataPointer.bindMemory(to: Float32.self, capacity: flat.count)
        for i in 0 ..< flat.count {
            ptr[i] = flat[i]
        }
        return out
    }

    static func mlMultiArrayFromNCHW(data: [Float32], height: Int, width: Int) throws -> MLMultiArray {
        let expectedCount = 3 * height * width
        guard data.count == expectedCount else {
            throw NSError(
                domain: "SamanthaRetouchEngine", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Invalid NCHW data count \(data.count), expected \(expectedCount)"]
            )
        }

        let out = try MLMultiArray(
            shape: [
                NSNumber(value: 1),
                NSNumber(value: 3),
                NSNumber(value: height),
                NSNumber(value: width),
            ],
            dataType: .float32
        )
        let ptr = out.dataPointer.bindMemory(to: Float32.self, capacity: expectedCount)
        for i in 0 ..< expectedCount {
            ptr[i] = data[i]
        }
        return out
    }

    static func nchwData(from output: MLMultiArray) -> (data: [Float32], height: Int, width: Int)? {
        guard output.shape.count == 4 else { return nil }
        let n = output.shape[0].intValue
        let c = output.shape[1].intValue
        let h = output.shape[2].intValue
        let w = output.shape[3].intValue
        guard n == 1, c == 3 else { return nil }
        let count = output.count
        let src = output.dataPointer.bindMemory(to: Float32.self, capacity: count)
        let data = Array(UnsafeBufferPointer(start: src, count: count))
        return (data, h, w)
    }

    @inline(__always)
    static func nchwIndex(channel: Int, y: Int, x: Int, height: Int, width: Int) -> Int {
        ((channel * height + y) * width) + x
    }

    static func buildIdentityLUTAtlasNCHW(dimension: Int) -> (data: [Float32], height: Int, width: Int) {
        let h = dimension
        let w = dimension * dimension
        var data = [Float32](repeating: 0, count: 3 * h * w)

        for b in 0 ..< dimension {
            for g in 0 ..< dimension {
                for r in 0 ..< dimension {
                    let x = b * dimension + r
                    let y = g
                    let rf = Float32(r) / Float32(dimension - 1)
                    let gf = Float32(g) / Float32(dimension - 1)
                    let bf = Float32(b) / Float32(dimension - 1)
                    data[nchwIndex(channel: 0, y: y, x: x, height: h, width: w)] = rf * 2.0 - 1.0
                    data[nchwIndex(channel: 1, y: y, x: x, height: h, width: w)] = gf * 2.0 - 1.0
                    data[nchwIndex(channel: 2, y: y, x: x, height: h, width: w)] = bf * 2.0 - 1.0
                }
            }
        }

        return (data, h, w)
    }

    static func apply3DLUTFromAtlas(
        atlasNCHW: [Float32],
        dimension: Int,
        sourceImage: CIImage
    ) -> CGImage? {
        let atlasHeight = dimension
        let atlasWidth = dimension * dimension
        guard atlasNCHW.count == 3 * atlasHeight * atlasWidth else { return nil }

        var cube = [Float32](repeating: 0, count: dimension * dimension * dimension * 4)
        for b in 0 ..< dimension {
            for g in 0 ..< dimension {
                for r in 0 ..< dimension {
                    let x = b * dimension + r
                    let y = g
                    let rr = atlasNCHW[nchwIndex(channel: 0, y: y, x: x, height: atlasHeight, width: atlasWidth)]
                    let gg = atlasNCHW[nchwIndex(channel: 1, y: y, x: x, height: atlasHeight, width: atlasWidth)]
                    let bb = atlasNCHW[nchwIndex(channel: 2, y: y, x: x, height: atlasHeight, width: atlasWidth)]

                    let lutBase = ((b * dimension + g) * dimension + r) * 4
                    cube[lutBase + 0] = max(0, min(1, rr))
                    cube[lutBase + 1] = max(0, min(1, gg))
                    cube[lutBase + 2] = max(0, min(1, bb))
                    cube[lutBase + 3] = 1.0
                }
            }
        }

        let cubeData = Data(bytes: cube, count: cube.count * MemoryLayout<Float32>.size)
        let extent = sourceImage.extent.integral
        let ctx = CIContext(options: nil)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)

        let outputImage: CIImage?
        if let filter = CIFilter(name: "CIColorCubeWithColorSpace") {
            filter.setValue(sourceImage, forKey: kCIInputImageKey)
            filter.setValue(dimension, forKey: "inputCubeDimension")
            filter.setValue(cubeData, forKey: "inputCubeData")
            filter.setValue(colorSpace, forKey: "inputColorSpace")
            outputImage = filter.outputImage
        } else if let filter = CIFilter(name: "CIColorCube") {
            filter.setValue(sourceImage, forKey: kCIInputImageKey)
            filter.setValue(dimension, forKey: "inputCubeDimension")
            filter.setValue(cubeData, forKey: "inputCubeData")
            outputImage = filter.outputImage
        } else {
            return nil
        }

        guard let outputImage else { return nil }
        return ctx.createCGImage(outputImage.cropped(to: extent), from: extent)
    }

    static func build3DLUTCubeDataFromAtlas(atlasNCHW: [Float32], dimension: Int) -> Data? {
        let atlasHeight = dimension
        let atlasWidth = dimension * dimension
        guard atlasNCHW.count == 3 * atlasHeight * atlasWidth else { return nil }

        var cube = [Float32](repeating: 0, count: dimension * dimension * dimension * 4)
        for b in 0 ..< dimension {
            for g in 0 ..< dimension {
                for r in 0 ..< dimension {
                    let x = b * dimension + r
                    let y = g
                    let rr = atlasNCHW[nchwIndex(channel: 0, y: y, x: x, height: atlasHeight, width: atlasWidth)]
                    let gg = atlasNCHW[nchwIndex(channel: 1, y: y, x: x, height: atlasHeight, width: atlasWidth)]
                    let bb = atlasNCHW[nchwIndex(channel: 2, y: y, x: x, height: atlasHeight, width: atlasWidth)]
                    let lutBase = ((b * dimension + g) * dimension + r) * 4
                    cube[lutBase + 0] = max(0, min(1, rr))
                    cube[lutBase + 1] = max(0, min(1, gg))
                    cube[lutBase + 2] = max(0, min(1, bb))
                    cube[lutBase + 3] = 1.0
                }
            }
        }
        return Data(bytes: cube, count: cube.count * MemoryLayout<Float32>.size)
    }

    static func padEdgeNCHW(
        _ input: [Float32],
        inputHeight: Int,
        inputWidth: Int,
        paddedHeight: Int,
        paddedWidth: Int
    ) -> [Float32] {
        var padded = [Float32](repeating: 0, count: 3 * paddedHeight * paddedWidth)
        for c in 0 ..< 3 {
            for y in 0 ..< paddedHeight {
                let srcY = min(y, inputHeight - 1)
                for x in 0 ..< paddedWidth {
                    let srcX = min(x, inputWidth - 1)
                    let srcIdx = nchwIndex(
                        channel: c, y: srcY, x: srcX, height: inputHeight, width: inputWidth)
                    let dstIdx = nchwIndex(
                        channel: c, y: y, x: x, height: paddedHeight, width: paddedWidth)
                    padded[dstIdx] = input[srcIdx]
                }
            }
        }
        return padded
    }

    static func extractPatchNCHW(
        from input: [Float32],
        sourceHeight: Int,
        sourceWidth: Int,
        originY: Int,
        originX: Int,
        patchSize: Int
    ) -> [Float32] {
        var patch = [Float32](repeating: 0, count: 3 * patchSize * patchSize)
        for c in 0 ..< 3 {
            for py in 0 ..< patchSize {
                for px in 0 ..< patchSize {
                    let srcIdx = nchwIndex(
                        channel: c,
                        y: originY + py,
                        x: originX + px,
                        height: sourceHeight,
                        width: sourceWidth
                    )
                    let dstIdx = nchwIndex(
                        channel: c, y: py, x: px, height: patchSize, width: patchSize)
                    patch[dstIdx] = input[srcIdx]
                }
            }
        }
        return patch
    }

    static func pastePatchNCHW(
        patch: [Float32],
        into output: inout [Float32],
        destinationHeight: Int,
        destinationWidth: Int,
        originY: Int,
        originX: Int,
        patchSize: Int
    ) {
        for c in 0 ..< 3 {
            for py in 0 ..< patchSize {
                for px in 0 ..< patchSize {
                    let srcIdx = nchwIndex(
                        channel: c, y: py, x: px, height: patchSize, width: patchSize)
                    let dstIdx = nchwIndex(
                        channel: c,
                        y: originY + py,
                        x: originX + px,
                        height: destinationHeight,
                        width: destinationWidth
                    )
                    output[dstIdx] = patch[srcIdx]
                }
            }
        }
    }

    static func cropNCHW(
        _ input: [Float32],
        inputHeight: Int,
        inputWidth: Int,
        outputHeight: Int,
        outputWidth: Int
    ) -> [Float32] {
        var output = [Float32](repeating: 0, count: 3 * outputHeight * outputWidth)
        for c in 0 ..< 3 {
            for y in 0 ..< outputHeight {
                for x in 0 ..< outputWidth {
                    let srcIdx = nchwIndex(
                        channel: c, y: y, x: x, height: inputHeight, width: inputWidth)
                    let dstIdx = nchwIndex(
                        channel: c, y: y, x: x, height: outputHeight, width: outputWidth)
                    output[dstIdx] = input[srcIdx]
                }
            }
        }
        return output
    }

    static func runRetouchDecoderTiled(
        model: MLModel,
        inputData: [Float32],
        inputHeight: Int,
        inputWidth: Int,
        retouchLatent: MLMultiArray,
        tileSize: Int
    ) throws -> [Float32]? {
        guard inputHeight > 0, inputWidth > 0, tileSize > 0 else { return nil }

        if inputHeight == tileSize && inputWidth == tileSize {
            let decoderInput = try mlMultiArrayFromNCHW(
                data: inputData, height: inputHeight, width: inputWidth)
            let decoderProvider = try MLDictionaryFeatureProvider(
                dictionary: [
                    "input_img": MLFeatureValue(multiArray: decoderInput),
                    "retouch_latent": MLFeatureValue(multiArray: retouchLatent),
                ])
            let decoderOutput = try model.prediction(from: decoderProvider)
            guard let outputImageArray = decoderOutput.featureValue(for: "output")?.multiArrayValue else {
                return nil
            }
            return nchwData(from: outputImageArray)?.data
        }

        let paddedHeight = ((inputHeight + tileSize - 1) / tileSize) * tileSize
        let paddedWidth = ((inputWidth + tileSize - 1) / tileSize) * tileSize
        let paddedInput = padEdgeNCHW(
            inputData,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            paddedHeight: paddedHeight,
            paddedWidth: paddedWidth
        )

        var stitched = [Float32](repeating: 0, count: 3 * paddedHeight * paddedWidth)

        for y in stride(from: 0, to: paddedHeight, by: tileSize) {
            for x in stride(from: 0, to: paddedWidth, by: tileSize) {
                let patch = extractPatchNCHW(
                    from: paddedInput,
                    sourceHeight: paddedHeight,
                    sourceWidth: paddedWidth,
                    originY: y,
                    originX: x,
                    patchSize: tileSize
                )
                let patchInput = try mlMultiArrayFromNCHW(data: patch, height: tileSize, width: tileSize)
                let outputPatch: [Float32]? = try autoreleasepool {
                    let decoderProvider = try MLDictionaryFeatureProvider(
                        dictionary: [
                            "input_img": MLFeatureValue(multiArray: patchInput),
                            "retouch_latent": MLFeatureValue(multiArray: retouchLatent),
                        ])
                    let decoderOutput = try model.prediction(from: decoderProvider)
                    guard
                        let outputImageArray = decoderOutput.featureValue(for: "output")?.multiArrayValue
                    else {
                        return nil
                    }
                    return nchwData(from: outputImageArray)?.data
                }
                guard let outputPatch else { return nil }
                pastePatchNCHW(
                    patch: outputPatch,
                    into: &stitched,
                    destinationHeight: paddedHeight,
                    destinationWidth: paddedWidth,
                    originY: y,
                    originX: x,
                    patchSize: tileSize
                )
            }
        }

        return cropNCHW(
            stitched,
            inputHeight: paddedHeight,
            inputWidth: paddedWidth,
            outputHeight: inputHeight,
            outputWidth: inputWidth
        )
    }

    static func runRetouchDecoderTiledFromCIImage(
        model: MLModel,
        sourceImage: CIImage,
        retouchLatent: MLMultiArray,
        tileSize: Int
    ) throws -> CGImage? {
        let extent = sourceImage.extent.integral
        let width = Int(extent.width)
        let height = Int(extent.height)
        guard width > 0, height > 0, tileSize > 0 else { return nil }

        let bytesPerPixel = 4
        let rowBytes = width * bytesPerPixel
        let outputCount = height * rowBytes

        // File-backed output buffer to reduce RAM pressure on very large images.
        let tempPath = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("samantha_retouch_\(UUID().uuidString).rgba")
        let fd = open(tempPath, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR)
        guard fd >= 0 else { return nil }
        if ftruncate(fd, off_t(outputCount)) != 0 {
            close(fd)
            tempPath.withCString { _ = unlink($0) }
            return nil
        }
        guard
            let outBase = mmap(
                nil,
                outputCount,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                0
            ),
            outBase != MAP_FAILED
        else {
            close(fd)
            tempPath.withCString { _ = unlink($0) }
            return nil
        }

        let outPtr = outBase.assumingMemoryBound(to: UInt8.self)
        memset(outBase, 0, outputCount)

        var ownershipTransferredToData = false
        func cleanupMapped() {
            munmap(outBase, outputCount)
            close(fd)
            tempPath.withCString { _ = unlink($0) }
        }
        defer {
            if !ownershipTransferredToData {
                cleanupMapped()
            }
        }

        let ctx = CIContext(options: nil)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        var srcRGBA = Data(count: outputCount)
        srcRGBA.withUnsafeMutableBytes { ptr in
            guard let base = ptr.baseAddress else { return }
            ctx.render(
                sourceImage,
                toBitmap: base,
                rowBytes: rowBytes,
                bounds: extent,
                format: .RGBA8,
                colorSpace: colorSpace
            )
        }

        for y in stride(from: 0, to: height, by: tileSize) {
            for x in stride(from: 0, to: width, by: tileSize) {
                let validH = min(tileSize, height - y)
                let validW = min(tileSize, width - x)
                var patch = [Float32](repeating: 0, count: 3 * tileSize * tileSize)
                for py in 0 ..< tileSize {
                    let sy = min(y + py, height - 1)
                    for px in 0 ..< tileSize {
                        let sx = min(x + px, width - 1)
                        let src = sy * rowBytes + sx * bytesPerPixel
                        let r = Float32(srcRGBA[src + 0]) / 255.0
                        let g = Float32(srcRGBA[src + 1]) / 255.0
                        let b = Float32(srcRGBA[src + 2]) / 255.0
                        let rIdx = ((0 * tileSize + py) * tileSize) + px
                        let gIdx = ((1 * tileSize + py) * tileSize) + px
                        let bIdx = ((2 * tileSize + py) * tileSize) + px
                        patch[rIdx] = r * 2.0 - 1.0
                        patch[gIdx] = g * 2.0 - 1.0
                        patch[bIdx] = b * 2.0 - 1.0
                    }
                }

                let patchInput = try mlMultiArrayFromNCHW(data: patch, height: tileSize, width: tileSize)
                let outputPatch: [Float32]? = try autoreleasepool {
                    let decoderProvider = try MLDictionaryFeatureProvider(
                        dictionary: [
                            "input_img": MLFeatureValue(multiArray: patchInput),
                            "retouch_latent": MLFeatureValue(multiArray: retouchLatent),
                        ])
                    let decoderOutput = try model.prediction(from: decoderProvider)
                    guard let outputImageArray = decoderOutput.featureValue(for: "output")?.multiArrayValue else {
                        return nil
                    }
                    return nchwData(from: outputImageArray)?.data
                }
                guard let patchOut = outputPatch else { return nil }

                for py in 0 ..< validH {
                    for px in 0 ..< validW {
                        let rIdx = ((0 * tileSize + py) * tileSize) + px
                        let gIdx = ((1 * tileSize + py) * tileSize) + px
                        let bIdx = ((2 * tileSize + py) * tileSize) + px
                        let r = UInt8(max(0, min(255, Int(patchOut[rIdx] * 255.0))))
                        let g = UInt8(max(0, min(255, Int(patchOut[gIdx] * 255.0))))
                        let b = UInt8(max(0, min(255, Int(patchOut[bIdx] * 255.0))))
                        let dst = ((y + py) * rowBytes) + ((x + px) * bytesPerPixel)
                        outPtr[dst + 0] = r
                        outPtr[dst + 1] = g
                        outPtr[dst + 2] = b
                        outPtr[dst + 3] = 255
                    }
                }
            }
        }

        let mappedData = Data(
            bytesNoCopy: outBase,
            count: outputCount,
            deallocator: .custom { _, _ in
                munmap(outBase, outputCount)
                close(fd)
                tempPath.withCString { _ = unlink($0) }
            }
        )
        ownershipTransferredToData = true
        let provider = CGDataProvider(data: mappedData as CFData)
        guard let provider else { return nil }
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue))
        guard let image = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: rowBytes,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else {
            return nil
        }
        return image
    }

    static func cgImage(fromNCHW output: MLMultiArray) -> CGImage? {
        guard output.shape.count == 4 else { return nil }
        let n = output.shape[0].intValue
        let c = output.shape[1].intValue
        let h = output.shape[2].intValue
        let w = output.shape[3].intValue
        guard n == 1, c == 3 else { return nil }

        let count = output.count
        let src = output.dataPointer.bindMemory(to: Float32.self, capacity: count)

        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        for y in 0 ..< h {
            for x in 0 ..< w {
                let rIdx = ((0 * c + 0) * h + y) * w + x
                let gIdx = ((0 * c + 1) * h + y) * w + x
                let bIdx = ((0 * c + 2) * h + y) * w + x

                let r = UInt8(max(0, min(255, Int(src[rIdx] * 255.0))))
                let g = UInt8(max(0, min(255, Int(src[gIdx] * 255.0))))
                let b = UInt8(max(0, min(255, Int(src[bIdx] * 255.0))))

                let dst = (y * w + x) * 4
                rgba[dst + 0] = r
                rgba[dst + 1] = g
                rgba[dst + 2] = b
                rgba[dst + 3] = 255
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue))
        let provider = CGDataProvider(data: Data(rgba) as CFData)
        return CGImage(
            width: w,
            height: h,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: w * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider!,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }

    static func cgImage(fromNCHWData data: [Float32], height: Int, width: Int) -> CGImage? {
        guard data.count == 3 * height * width else { return nil }

        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let rIdx = nchwIndex(channel: 0, y: y, x: x, height: height, width: width)
                let gIdx = nchwIndex(channel: 1, y: y, x: x, height: height, width: width)
                let bIdx = nchwIndex(channel: 2, y: y, x: x, height: height, width: width)

                let r = UInt8(max(0, min(255, Int(data[rIdx] * 255.0))))
                let g = UInt8(max(0, min(255, Int(data[gIdx] * 255.0))))
                let b = UInt8(max(0, min(255, Int(data[bIdx] * 255.0))))

                let dst = (y * width + x) * 4
                rgba[dst + 0] = r
                rgba[dst + 1] = g
                rgba[dst + 2] = b
                rgba[dst + 3] = 255
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue))
        let provider = CGDataProvider(data: Data(rgba) as CFData)
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider!,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }
}
