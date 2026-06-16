import CoreImage
import VeraRetouchCore
import MLXLMCommon
#if canImport(Photos)
import Photos
#endif
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

#if canImport(UIKit)
import UIKit
import AVFoundation
typealias PlatformImage = UIImage
#elseif canImport(AppKit)
import AppKit
typealias PlatformImage = NSImage
#endif

private enum AppTypography {
    static var regularName: String {
        #if os(iOS)
        if UIFont(name: "AvenirNext-Regular", size: 14) != nil { return "AvenirNext-Regular" }
        #elseif os(macOS)
        if NSFont(name: "AvenirNext-Regular", size: 14) != nil { return "AvenirNext-Regular" }
        #endif
        return "HelveticaNeue"
    }

    static var semiboldName: String {
        #if os(iOS)
        if UIFont(name: "AvenirNext-DemiBold", size: 14) != nil { return "AvenirNext-DemiBold" }
        #elseif os(macOS)
        if NSFont(name: "AvenirNext-DemiBold", size: 14) != nil { return "AvenirNext-DemiBold" }
        #endif
        return regularName
    }

    static var boldName: String {
        #if os(iOS)
        if UIFont(name: "AvenirNext-Bold", size: 14) != nil { return "AvenirNext-Bold" }
        #elseif os(macOS)
        if NSFont(name: "AvenirNext-Bold", size: 14) != nil { return "AvenirNext-Bold" }
        #endif
        return semiboldName
    }

    static func font(size: CGFloat, weight: Font.Weight = .regular) -> Font {
        let face: String
        switch weight {
        case .bold, .heavy, .black:
            face = boldName
        case .semibold, .medium:
            face = semiboldName
        default:
            face = regularName
        }
        return .custom(face, size: size)
    }
}

// MARK: - Image helpers
extension PlatformImage {
    func resized(to size: CGSize) -> PlatformImage? {
        #if os(iOS)
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: size))
        }
        #elseif os(macOS)
        let newImage = NSImage(size: size)
        newImage.lockFocus()
        let fromRect = NSRect(origin: .zero, size: self.size)
        let destRect = NSRect(origin: .zero, size: size)
        self.draw(in: destRect, from: fromRect, operation: .copy, fraction: 1.0)
        newImage.unlockFocus()
        return newImage
        #endif
    }

    #if os(iOS)
    /// Bake EXIF orientation into pixel data so downstream CoreImage/CoreML
    /// sees an upright image.
    func normalizedUp() -> UIImage {
        if imageOrientation == .up { return self }
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = scale
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: size))
        }
    }
    #endif
}

private extension View {
    func glassButton(cornerRadius: CGFloat = 12) -> some View {
        self
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .opacity(0.92)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .stroke(
                        LinearGradient(
                            colors: [Color.white.opacity(0.95), Color.white.opacity(0.55)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 1.25
                    )
            )
            .shadow(color: Color.black.opacity(0.06), radius: 8, x: 0, y: 4)
    }
}

private extension View {
    func glassCard(cornerRadius: CGFloat = 14) -> some View {
        self
            .padding(10)
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .opacity(0.88)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .stroke(Color.white.opacity(0.42), lineWidth: 0.85)
            )
    }
}

private extension View {
    func transparentListRow(horizontal: CGFloat = 12, vertical: CGFloat = 8) -> some View {
        self
            .listRowBackground(Color.clear)
            .listRowSeparator(.hidden)
            .listSectionSeparator(.hidden)
            .listRowInsets(EdgeInsets(top: vertical, leading: horizontal, bottom: vertical, trailing: horizontal))
    }
}

struct ContentView: View {
    private static let autoRetouchPrompt =
        "<Auto_Retouch_Task>\nNow, you are acting as a Retouch Agent. When I provide an image, please state the problems found in the image (from 3 aspects: lighting, global_color, specific color), and give the solution and retouch tokens."
    private static let styleRetouchPrompt =
        "<Style_Retouch_Task>\nNow, you are acting as a Retouch Agent. I will provide an image and an instruction, please give me a retouch plan and retouch tokens.\nInstruction:"
    private static let defaultStyleInstruction = "I want a Dreamy Pink style."
    private static let stylePresetTags: [String] = [
        "Dreamy Pink",
        "Dark Movie",
        "Autumn",
        "Cinematic",
        "Green Orange",
        "Retro",
        "Vintage Film",
        "Documentary Cinematic",
        "High-End Grey",
    ]
    private static let professionalRetouchPrompt =
        "<Professional_Retouch_Task>\nNow, you are acting as a Retouch Agent. I will provide an image and an professional instruction (plain text description or retouch operator parameters range from -1.0 to 1.0), please give me a retouch plan and retouch tokens.\nInstruction:"
    private let minImageHeight: CGFloat = 180
    @State private var model = VeraRetouchModel()

    @State private var selectedMode: String = "Auto"
    @State private var uploadedImage: PlatformImage?
    @State private var selectedPickerItem: PhotosPickerItem?
    @State private var sourceCIImage: CIImage?
    @State private var referenceBeforeImage: PlatformImage?
    @State private var referenceAfterImage: PlatformImage?
    @State private var referenceBeforeCIImage: CIImage?
    @State private var referenceAfterCIImage: CIImage?
    @State private var selectedReferenceBeforePickerItem: PhotosPickerItem?
    @State private var selectedReferenceAfterPickerItem: PhotosPickerItem?
    @State private var showSaveAlert: Bool = false
    @State private var saveAlertMessage: String = ""
    @State private var selectedLUTMode: String = "33"
    @State private var samplingTemperature: Double = 0.2
    @State private var samplingTopP: Double = 0.8
    @State private var showSamplingInfoHint: Bool = false
    @State private var showLUTInfoHint: Bool = false
    @State private var autoEnableLight: Bool = true
    @State private var autoEnableGlobalColor: Bool = true
    @State private var autoEnableSpecificColor: Bool = true
    @State private var autoRunRequested: Bool = false
    @State private var hasAutoRunResult: Bool = false
    @State private var autoFollowOutput: Bool = false
    @State private var showFakeProgress: Bool = false
    @State private var fakeProgressValue: Double = 0
    @State private var fakeProgressTask: Task<Void, Never>?
    @State private var showWelcomeOverlay: Bool = true
    @State private var welcomeFloatUp: Bool = false
    @State private var welcomeTextVisible: Bool = false
    @State private var isManualRerendering: Bool = false
    @State private var lastOutputChangeAt: Date?
    @State private var sawOutputInCurrentRun: Bool = false
    private let autoLUTPixelThreshold: Int = 1920 * 1080
    @State private var isLivePhotoInput: Bool = false
    #if os(iOS)
    @State private var showLivePhotoPickerSheet: Bool = false
    @State private var livePhotoAsset: PHAsset?
    @State private var livePhotoVideoURL: URL?
    @State private var inputLivePhoto: PHLivePhoto?
    @State private var outputLivePhoto: PHLivePhoto?
    @State private var outputLivePhotoPhotoURL: URL?
    @State private var outputLivePhotoVideoURL: URL?
    @State private var livePhotoDurationSeconds: Double = 0
    @State private var livePhotoFrameProgress: Double = 0.5
    @State private var livePhotoPreviewTask: Task<Void, Never>?
    @State private var livePhotoFrameRequestID: Int = 0
    @State private var livePhotoReferenceFrameImage: UIImage?
    @State private var livePhotoSyncReplayToken: Int = 0
    @State private var inputLivePhotoReplayToken: Int = 0
    @State private var outputLivePhotoReplayToken: Int = 0
    @State private var inputLivePhotoViewID: Int = 0
    @State private var outputLivePhotoViewID: Int = 0
    @State private var isInputLivePhotoPlaying: Bool = false
    @State private var isSyncLivePhotoPlaying: Bool = false
    @FocusState private var styleEditorFocused: Bool
    #endif

    private var preferredSectionHeaderFont: Font {
        AppTypography.font(size: 16, weight: .semibold)
    }

    private var shouldShowComparisonSection: Bool {
        #if os(iOS)
        if isLivePhotoInput {
            return outputLivePhoto != nil
        }
        #endif
        return model.retouchedImage != nil
    }

    #if os(iOS)
    private let livePhotoPreviewHeight: CGFloat = 240
    #endif

    @ViewBuilder
    private func sectionHeader(_ title: String) -> some View {
        Text(title)
            .font(preferredSectionHeaderFont)
            .foregroundStyle(Color.primary.opacity(0.82))
            .tracking(0.3)
            .textCase(nil)
    }

    @State private var styleInstruction: String = Self.defaultStyleInstruction
    @State private var styleInstructionIsDefault: Bool = true
    @State private var selectedStyleTag: String? = "Dreamy Pink"

    init() {
        #if os(iOS)
        UITableView.appearance().backgroundColor = .clear
        UITableViewCell.appearance().backgroundColor = .clear
        UITableViewHeaderFooterView.appearance().tintColor = .clear
        UICollectionView.appearance().backgroundColor = .clear
        UICollectionReusableView.appearance().backgroundColor = .clear
        if #available(iOS 14.0, *) {
            UICollectionViewListCell.appearance().backgroundConfiguration = UIBackgroundConfiguration.clear()
        }
        #endif
    }

    @ViewBuilder
    private var atmosphericBackground: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.95, green: 0.965, blue: 0.99),
                    Color(red: 0.91, green: 0.93, blue: 0.97),
                    Color(red: 0.96, green: 0.955, blue: 0.985),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            Circle()
                .fill(Color(red: 0.50, green: 0.70, blue: 1.0).opacity(0.22))
                .frame(width: 260, height: 260)
                .blur(radius: 28)
                .offset(x: -130, y: -320)

            Circle()
                .fill(Color(red: 0.72, green: 0.90, blue: 0.98).opacity(0.18))
                .frame(width: 240, height: 240)
                .blur(radius: 30)
                .offset(x: 150, y: -220)

            Circle()
                .fill(Color(red: 0.80, green: 0.78, blue: 1.0).opacity(0.12))
                .frame(width: 300, height: 300)
                .blur(radius: 40)
                .offset(x: 120, y: 360)
        }
        .ignoresSafeArea()
    }

    @State private var lightParams: [String: Double] = [
        "Exposure2012": 0.0,
        "Contrast2012": 0.0,
        "Highlights2012": 0.0,
        "Shadows2012": 0.0,
        "Whites2012": 0.0,
        "Blacks2012": 0.0,
        "ParametricShadows": 0.0,
        "ParametricDarks": 0.0,
        "ParametricLights": 0.0,
        "ParametricHighlights": 0.0,
    ]
    private let lightKeys = [
        "Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012",
        "ParametricShadows", "ParametricDarks", "ParametricLights", "ParametricHighlights",
    ]

    @State private var tempParams: [String: Double] = [
        "IncrementalTemperature": 0.0, "IncrementalTint": 0.0, "Vibrance": 0.0, "Saturation": 0.0
    ]
    private let tempKeys = ["IncrementalTemperature", "IncrementalTint", "Vibrance", "Saturation"]

    @State private var mixerParams: [String: Double] = [
        "HueAdjustmentRed": 0.0, "HueAdjustmentOrange": 0.0, "HueAdjustmentYellow": 0.0,
        "HueAdjustmentGreen": 0.0, "HueAdjustmentAqua": 0.0, "HueAdjustmentBlue": 0.0,
        "HueAdjustmentPurple": 0.0, "HueAdjustmentMagenta": 0.0,
        "SaturationAdjustmentRed": 0.0, "SaturationAdjustmentOrange": 0.0, "SaturationAdjustmentYellow": 0.0,
        "SaturationAdjustmentGreen": 0.0, "SaturationAdjustmentAqua": 0.0, "SaturationAdjustmentBlue": 0.0,
        "SaturationAdjustmentPurple": 0.0, "SaturationAdjustmentMagenta": 0.0,
        "LuminanceAdjustmentRed": 0.0, "LuminanceAdjustmentOrange": 0.0, "LuminanceAdjustmentYellow": 0.0,
        "LuminanceAdjustmentGreen": 0.0, "LuminanceAdjustmentAqua": 0.0, "LuminanceAdjustmentBlue": 0.0,
        "LuminanceAdjustmentPurple": 0.0, "LuminanceAdjustmentMagenta": 0.0,
    ]
    private let mixerKeys: [String] = [
        "HueAdjustmentRed", "HueAdjustmentOrange", "HueAdjustmentYellow", "HueAdjustmentGreen",
        "HueAdjustmentAqua", "HueAdjustmentBlue", "HueAdjustmentPurple", "HueAdjustmentMagenta",
        "SaturationAdjustmentRed", "SaturationAdjustmentOrange", "SaturationAdjustmentYellow",
        "SaturationAdjustmentGreen", "SaturationAdjustmentAqua", "SaturationAdjustmentBlue",
        "SaturationAdjustmentPurple", "SaturationAdjustmentMagenta", "LuminanceAdjustmentRed",
        "LuminanceAdjustmentOrange", "LuminanceAdjustmentYellow", "LuminanceAdjustmentGreen",
        "LuminanceAdjustmentAqua", "LuminanceAdjustmentBlue", "LuminanceAdjustmentPurple",
        "LuminanceAdjustmentMagenta",
    ]

    @ViewBuilder
    private var modePicker: some View {
        Picker("Mode", selection: $selectedMode) {
            Text("Auto-Retouch").tag("Auto")
            Text("Style-Retouch").tag("Style")
            Text("Param-Retouch").tag("Professional")
            Text("Ref-Retouch").tag("Reference")
        }
        .pickerStyle(.segmented)
        .padding(.horizontal, 12)
        #if os(iOS)
        .padding(.top, 6)
        #endif
        .padding(.bottom, 8)
        .disabled(model.running)
    }

    @ViewBuilder
    private var topGlassHeader: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Image("SloganLogo")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 50, height: 50)
                Text("VeraRetouch")
                    .font(AppTypography.font(size: 20, weight: .bold))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [
                                Color(red: 0.12, green: 0.47, blue: 0.98),
                                Color(red: 0.15, green: 0.78, blue: 0.57),
                                Color(red: 0.98, green: 0.57, blue: 0.18),
                            ],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
            }
            .padding(.top, 6)
            .padding(.bottom, 4)

            modePicker
        }
        .padding(.horizontal, 12)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(.ultraThinMaterial)
                .opacity(0.88)
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .stroke(
                            LinearGradient(
                                colors: [Color.white.opacity(0.82), Color.white.opacity(0.32)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1.0
                        )
                )
        )
        .shadow(color: Color.black.opacity(0.06), radius: 10, x: 0, y: 5)
        .padding(.horizontal, 8)
        .padding(.bottom, 8)
    }

    @ViewBuilder
    private func contentList(scrollProxy: ScrollViewProxy) -> some View {
        List {
            inputImageSection
            fixedPromptSection
            referenceRetouchSection
            livePhotoReferenceSection
            lutSection
            samplingSection
            styleInstructionSection
            professionalAdjustmentSections
            runSection
            analysisResultSection
            comparisonSection
            autoAdjustmentsSection
            listBottomAnchor
        }
        #if os(iOS)
        .listStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Color.clear)
        .listRowSeparator(.hidden)
        .listSectionSeparator(.hidden)
        .listRowBackground(Color.clear)
        .simultaneousGesture(
            TapGesture().onEnded {
                dismissKeyboard()
            }
        )
        .simultaneousGesture(
            DragGesture(minimumDistance: 4).onChanged { _ in
                if model.running {
                    autoFollowOutput = false
                }
            }
        )
        #else
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
        .background(Color.clear)
        .listRowSeparator(.hidden)
        .listSectionSeparator(.hidden)
        .listRowBackground(Color.clear)
        #endif
        .onChange(of: selectedMode) { _, _ in
            #if os(iOS)
            dismissKeyboard()
            #endif
            clearModeState()
            withAnimation {
                scrollProxy.scrollTo("TopAnchor", anchor: .top)
            }
        }
        .onChange(of: styleInstruction) { _, newValue in
            guard let selectedStyleTag else { return }
            if newValue != stylePromptText(for: selectedStyleTag) {
                self.selectedStyleTag = nil
            }
        }
        .onChange(of: autoEnableLight) { _, _ in
            handleAutoMaskToggleChanged()
        }
        .onChange(of: autoEnableGlobalColor) { _, _ in
            handleAutoMaskToggleChanged()
        }
        .onChange(of: autoEnableSpecificColor) { _, _ in
            handleAutoMaskToggleChanged()
        }
        .onChange(of: model.output) { _, newOutput in
            if model.running {
                if !newOutput.isEmpty {
                    sawOutputInCurrentRun = true
                    lastOutputChangeAt = Date()
                }
                if showFakeProgress {
                    stopFakeProgressImmediately()
                }
            }
            guard autoFollowOutput, !newOutput.isEmpty else { return }
            withAnimation(.linear(duration: 0.12)) {
                scrollProxy.scrollTo("ListBottomAnchor", anchor: .bottom)
            }
        }
        .onReceive(Timer.publish(every: 0.2, on: .main, in: .common).autoconnect()) { _ in
            if model.running, model.isStreamingOutput, autoFollowOutput, !model.output.isEmpty {
                withAnimation(.linear(duration: 0.12)) {
                    scrollProxy.scrollTo("ListBottomAnchor", anchor: .bottom)
                }
            }

            // Show centered fake progress only after text output has gone quiet,
            // which approximates the heavy image render stage.
            if model.running,
                sawOutputInCurrentRun,
                !model.isStreamingOutput,
                model.retouchedImage == nil
            {
                if !showFakeProgress {
                    startFakeProgress()
                }
            }
        }
        .onChange(of: model.running) { _, isRunning in
            if !isRunning {
                autoFollowOutput = false
                if !isManualRerendering {
                    completeFakeProgress()
                }
                if autoRunRequested {
                    hasAutoRunResult = model.hasCachedRetouchState
                    autoRunRequested = false
                }
            }
        }
    }

    @ViewBuilder
    private var inputImageSection: some View {
        Section {
            #if os(iOS)
            VStack(spacing: 10) {
                if isLivePhotoInput, let inputLP = inputLivePhoto {
                    LivePhotoPreviewView(
                        livePhoto: inputLP,
                        replayToken: inputLivePhotoReplayToken,
                        autoPlay: false
                    )
                    .id("input-live-\(inputLivePhotoViewID)")
                    .frame(maxWidth: .infinity)
                    .frame(height: livePhotoPreviewHeight)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))

                    HStack {
                        Spacer()
                        Button {
                            playInputLivePhotoPreview()
                        } label: {
                            Label("Play", systemImage: "play.fill")
                                .font(AppTypography.font(size: 14, weight: .semibold))
                                .frame(width: 120, alignment: .center)
                                .foregroundStyle(Color.accentColor)
                                .glassButton(cornerRadius: 10)
                        }
                        .buttonStyle(.plain)
                        .opacity(isInputLivePhotoPlaying ? 0.45 : 1.0)
                        .disabled(isInputLivePhotoPlaying)
                        Spacer()
                    }
                } else {
                    ImageContainerView(platformImage: uploadedImage, cgImage: nil, minHeight: minImageHeight) {
                        VStack(spacing: 8) {
                            Image(systemName: "photo.badge.plus")
                                .font(AppTypography.font(size: 28, weight: .semibold))
                            Text("No image selected")
                        }
                    }
                }

                                HStack(spacing: 10) {
                                    PhotosPicker(selection: $selectedPickerItem, matching: .images) {
                                        HStack(spacing: 8) {
                                            Image(systemName: "photo")
                                                .frame(width: 16, alignment: .center)
                                            Text("Select from Photos")
                                                .lineLimit(1)
                                        }
                                        .font(AppTypography.font(size: 13, weight: .semibold))
                                        .foregroundColor(.accentColor)
                                        .frame(maxWidth: .infinity, alignment: .center)
                                        .glassButton(cornerRadius: 10)
                                    }
                                    .frame(maxWidth: .infinity)
                                    .buttonStyle(.plain)

                                    Button {
                                        showLivePhotoPickerSheet = true
                                    } label: {
                                        HStack(spacing: 8) {
                                            Image(systemName: "livephoto")
                                                .frame(width: 16, alignment: .center)
                                            Text("Select Live Photo")
                                                .lineLimit(1)
                                        }
                                        .font(AppTypography.font(size: 13, weight: .semibold))
                                        .foregroundColor(.accentColor)
                                        .frame(maxWidth: .infinity, alignment: .center)
                                        .glassButton(cornerRadius: 10)
                                    }
                                    .frame(maxWidth: .infinity)
                                    .buttonStyle(.plain)
                                }
            }
            .onChange(of: selectedPickerItem) { newItem in
                Task {
                    guard let item = newItem else { return }
                    dismissKeyboard()
                    clearInferenceOutputsOnly()

                    if let localId = item.itemIdentifier {
                        let assets = PHAsset.fetchAssets(withLocalIdentifiers: [localId], options: nil)
                        if let asset = assets.firstObject, asset.mediaSubtypes.contains(.photoLive) {
                            await prepareLivePhotoInput(from: asset)
                            return
                        }
                    }

                    guard let data = try? await item.loadTransferable(type: Data.self) else {
                        return
                    }
                    guard let uiImage = UIImage(data: data) else { return }
                    let normalized = uiImage.normalizedUp()
                    uploadedImage = normalized
                    sourceCIImage = CIImage(image: normalized)
                    isLivePhotoInput = false
                    selectedLUTMode = "33"
                    livePhotoAsset = nil
                    livePhotoVideoURL = nil
                    inputLivePhoto = nil
                    outputLivePhoto = nil
                    outputLivePhotoPhotoURL = nil
                    outputLivePhotoVideoURL = nil
                    livePhotoDurationSeconds = 0
                    livePhotoFrameProgress = 0.5
                    livePhotoFrameRequestID = 0
                    livePhotoReferenceFrameImage = nil
                    livePhotoSyncReplayToken = 0
                    isInputLivePhotoPlaying = false
                    isSyncLivePhotoPlaying = false
                    livePhotoPreviewTask?.cancel()
                    livePhotoPreviewTask = nil
                    if let image = sourceCIImage {
                        applyAutoRendererModeIfNeeded(for: image)
                    }
                }
            }
            #elseif os(macOS)
            Button(action: { selectImageMac() }) {
                ImageContainerView(platformImage: uploadedImage, cgImage: nil, minHeight: minImageHeight) {
                    VStack(spacing: 8) {
                        Image(systemName: "folder.fill")
                            .font(AppTypography.font(size: 28, weight: .semibold))
                        Text("Click to Select Image")
                    }
                }
            }
            .buttonStyle(.plain)
            #endif
        } header: {
            sectionHeader("Input Image")
        }
        .transparentListRow(horizontal: 10, vertical: 6)
        .id("TopAnchor")
    }

    @ViewBuilder
    private var fixedPromptSection: some View {
        if selectedMode != "Reference" {
            Section {
                Text(fixedPromptForCurrentMode)
                    .font(AppTypography.font(size: 11))
                    .foregroundColor(.secondary)
                    .textSelection(.enabled)
                    .glassCard(cornerRadius: 12)
            } header: {
                sectionHeader("Fixed Prompt (Read Only)")
            }
            .transparentListRow()
        }
    }

    @ViewBuilder
    private var referenceRetouchSection: some View {
        if selectedMode == "Reference" {
            Section {
                VStack(alignment: .leading, spacing: 12) {
                    referenceImagePicker(
                        title: "Reference Before",
                        systemImage: "photo",
                        image: referenceBeforeImage,
                        pickerItem: $selectedReferenceBeforePickerItem,
                        kind: "before"
                    )

                    referenceImagePicker(
                        title: "Reference After",
                        systemImage: "photo.fill",
                        image: referenceAfterImage,
                        pickerItem: $selectedReferenceAfterPickerItem,
                        kind: "after"
                    )
                }
                .glassCard(cornerRadius: 12)
                .onChange(of: selectedReferenceBeforePickerItem) { _, newItem in
                    Task { await loadReferencePickerItem(newItem, kind: "before") }
                }
                .onChange(of: selectedReferenceAfterPickerItem) { _, newItem in
                    Task { await loadReferencePickerItem(newItem, kind: "after") }
                }
            } header: {
                sectionHeader("Reference Retouch")
            }
            .transparentListRow()
        }
    }

    @ViewBuilder
    private func referenceImagePicker(
        title: String,
        systemImage: String,
        image: PlatformImage?,
        pickerItem: Binding<PhotosPickerItem?>,
        kind: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(title, systemImage: systemImage)
                .font(AppTypography.font(size: 12, weight: .semibold))
                .foregroundColor(.secondary)
            ImageContainerView(platformImage: image, cgImage: nil, minHeight: 130) {
                VStack(spacing: 8) {
                    Image(systemName: systemImage)
                        .font(AppTypography.font(size: 24, weight: .semibold))
                    Text("Select Image")
                }
            }
            #if os(iOS)
            PhotosPicker(selection: pickerItem, matching: .images) {
                Label("Select", systemImage: "photo")
                    .font(AppTypography.font(size: 13, weight: .semibold))
                    .frame(maxWidth: .infinity, alignment: .center)
                    .foregroundColor(.accentColor)
                    .glassButton(cornerRadius: 10)
            }
            .buttonStyle(.plain)
            .disabled(model.running)
            #elseif os(macOS)
            Button {
                selectReferenceImageMac(kind: kind)
            } label: {
                Label("Select", systemImage: "folder.fill")
                    .font(AppTypography.font(size: 13, weight: .semibold))
                    .frame(maxWidth: .infinity, alignment: .center)
                    .foregroundColor(.accentColor)
                    .glassButton(cornerRadius: 10)
            }
            .buttonStyle(.plain)
            .disabled(model.running)
            #endif
        }
    }

    @ViewBuilder
    private var livePhotoReferenceSection: some View {
        #if os(iOS)
        if isLivePhotoInput, livePhotoVideoURL != nil {
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    ImageContainerView(
                        platformImage: livePhotoReferenceFrameImage ?? uploadedImage,
                        cgImage: nil,
                        minHeight: minImageHeight
                    ) {
                        Text("Reference frame preview unavailable")
                    }
                    Text("Drag to pick a reference frame")
                        .font(AppTypography.font(size: 12))
                        .foregroundColor(.secondary)
                    Slider(value: $livePhotoFrameProgress, in: 0 ... 1, step: 0.001)
                        .onChange(of: livePhotoFrameProgress) { _, newValue in
                            updateLivePhotoReferenceFrame(progress: newValue)
                        }
                    Text("\(formattedTimestamp(livePhotoDurationSeconds * livePhotoFrameProgress)) / \(formattedTimestamp(livePhotoDurationSeconds))")
                        .font(AppTypography.font(size: 11))
                        .foregroundColor(.secondary)
                }
                .glassCard(cornerRadius: 12)
            } header: {
                sectionHeader("Live Photo Reference Frame")
            }
            .transparentListRow()
        }
        #endif
    }

    @ViewBuilder
    private var lutSection: some View {
        Section {
            Picker("LUT Mode", selection: $selectedLUTMode) {
                Text("Off").tag("Off")
                Text("33x33x33").tag("33")
                Text("65x65x65").tag("65")
            }
            .pickerStyle(.segmented)
            .disabled(model.running)
            .glassCard(cornerRadius: 12)
        } header: {
            HStack(spacing: 8) {
                sectionHeader("LUT Acceleration")
                ZStack(alignment: .topLeading) {
                    Button {
                        withAnimation(.easeInOut(duration: 0.12)) {
                            showLUTInfoHint.toggle()
                            if showLUTInfoHint { showSamplingInfoHint = false }
                        }
                    } label: {
                        Image(systemName: "info.circle.fill")
                            .foregroundColor(.accentColor)
                            .font(AppTypography.font(size: 13))
                            .frame(width: 28, height: 28)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)

                    if showLUTInfoHint {
                        Text("LUT acceleration can reduce memory and speed up rendering, but may slightly reduce precision. It is automatically enabled when the uploaded image is larger than 1080p.")
                            .font(AppTypography.font(size: 11))
                            .foregroundColor(.primary)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 6)
                            .background(
                                RoundedRectangle(cornerRadius: 8, style: .continuous)
                                    .fill(.ultraThinMaterial)
                                    .opacity(0.78)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                                            .stroke(Color.white.opacity(0.42), lineWidth: 0.8)
                                    )
                            )
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: 280, alignment: .leading)
                            .offset(x: 18, y: -10)
                            .transition(.opacity)
                            .zIndex(1)
                    }
                }
                Spacer()
            }
        }
        .transparentListRow()
    }

    @ViewBuilder
    private var samplingSection: some View {
        if selectedMode == "Auto" || selectedMode == "Style" {
            Section {
                VStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Temperature")
                            Spacer()
                            Text(String(format: "%.2f", samplingTemperature))
                                .font(AppTypography.font(size: 12))
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $samplingTemperature, in: 0.0 ... 1.0, step: 0.01)
                            .disabled(model.running)
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Top-p")
                            Spacer()
                            Text(String(format: "%.2f", samplingTopP))
                                .font(AppTypography.font(size: 12))
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $samplingTopP, in: 0.0 ... 1.0, step: 0.01)
                            .disabled(model.running)
                    }
                }
                .glassCard(cornerRadius: 12)
            } header: {
                HStack(spacing: 8) {
                    sectionHeader("Sampling")
                    ZStack(alignment: .topLeading) {
                        Button {
                            withAnimation(.easeInOut(duration: 0.12)) {
                                showSamplingInfoHint.toggle()
                                if showSamplingInfoHint { showLUTInfoHint = false }
                            }
                        } label: {
                            Image(systemName: "info.circle.fill")
                                .foregroundColor(.accentColor)
                                .font(AppTypography.font(size: 13))
                                .frame(width: 28, height: 28)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)

                        if showSamplingInfoHint {
                            Text("Temperature controls randomness; Top-p controls token pool size (higher values are more diverse).")
                                .font(AppTypography.font(size: 11))
                                .foregroundColor(.primary)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 6)
                                .background(
                                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                                        .fill(.ultraThinMaterial)
                                        .opacity(0.78)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                                .stroke(Color.white.opacity(0.42), lineWidth: 0.8)
                                        )
                                )
                                .fixedSize(horizontal: false, vertical: true)
                                .frame(maxWidth: 260, alignment: .leading)
                                .offset(x: 18, y: -10)
                                .transition(.opacity)
                                .zIndex(1)
                        }
                    }
                    Spacer()
                }
            }
            .transparentListRow()
        }
    }

    @ViewBuilder
    private var styleInstructionSection: some View {
        if selectedMode == "Style" {
            Section {
                #if os(iOS)
                StylePromptTextView(
                    text: $styleInstruction,
                    isDefault: $styleInstructionIsDefault,
                    defaultText: Self.defaultStyleInstruction,
                    fontName: AppTypography.regularName
                )
                .frame(height: 110)
                .overlay(
                    RoundedRectangle(cornerRadius: 5)
                        .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
                )
                #else
                TextEditor(text: $styleInstruction)
                    .frame(height: 110)
                    .font(AppTypography.font(size: 12))
                    .foregroundStyle(styleInstructionIsDefault ? .secondary : .primary)
                    .overlay(
                        RoundedRectangle(cornerRadius: 5)
                            .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
                    )
                #endif

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(Self.stylePresetTags, id: \.self) { tag in
                            Button {
                                applyStylePresetTag(tag)
                            } label: {
                                Text(tag)
                                    .font(AppTypography.font(size: 12, weight: .medium))
                                    .foregroundStyle(selectedStyleTag == tag ? Color.white : Color.primary)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                                    .background(
                                        Capsule(style: .continuous)
                                            .fill(selectedStyleTag == tag ? Color.accentColor : Color.gray.opacity(0.18))
                                    )
                            }
                            .buttonStyle(.plain)
                            .disabled(model.running)
                        }
                    }
                    .padding(.vertical, 2)
                }
                .glassCard(cornerRadius: 12)
            } header: {
                sectionHeader("Style Instruction")
            }
            .transparentListRow()
        }
    }

    @ViewBuilder
    private var professionalAdjustmentSections: some View {
        if selectedMode == "Professional" {
            Section {
                ForEach(lightKeys, id: \.self) { key in
                    CleanParamRow(key: key, params: $lightParams, range: -100.0 ... 100.0)
                }
                .glassCard(cornerRadius: 12)
            } header: {
                sectionHeader("Light Adjustments")
            }
            .transparentListRow()

            Section {
                ForEach(tempKeys, id: \.self) { key in
                    CleanParamRow(key: key, params: $tempParams, range: -100.0 ... 100.0)
                }
                .glassCard(cornerRadius: 12)
            } header: {
                sectionHeader("Color Temperature")
            }
            .transparentListRow()

            Section {
                ForEach(mixerKeys, id: \.self) { key in
                    CleanParamRow(key: key, params: $mixerParams, range: -100.0 ... 100.0)
                }
                .glassCard(cornerRadius: 12)
            } header: {
                sectionHeader("Color Mixer")
            }
            .transparentListRow()

            Section {
                HStack {
                    Spacer()
                    Button {
                        resetAllProfessionalParams()
                    } label: {
                        Text("RESET ALL")
                            .font(AppTypography.font(size: 13, weight: .semibold))
                            .frame(width: 170, alignment: .center)
                            .foregroundStyle(Color.accentColor)
                            .glassButton()
                    }
                    .buttonStyle(.plain)
                    .disabled(model.running)
                    Spacer()
                }
                .glassCard(cornerRadius: 12)
            }
            .transparentListRow()
        }
    }

    @ViewBuilder
    private var runSection: some View {
        Section {
            HStack {
                Button {
                    #if os(iOS)
                    dismissKeyboard()
                    #endif
                    autoFollowOutput = true
                    runAnalysis()
                } label: {
                    Text(runButtonTitle)
                        .frame(maxWidth: .infinity, minHeight: 42)
                        .font(AppTypography.font(size: 17, weight: .semibold))
                        .foregroundColor(.accentColor)
                        .background(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .fill(Color.clear)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .stroke(Color.white.opacity(0.70), lineWidth: 1.25)
                        )
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(!canRunCurrentMode)
                Spacer(minLength: 0)
            }
        }
        .transparentListRow()
    }

    @ViewBuilder
    private var analysisResultSection: some View {
        if !model.output.isEmpty {
            Section {
                Text(model.output)
                    .font(AppTypography.font(size: 9))
                    .lineSpacing(3)
                    .textSelection(.enabled)
                    .glassCard(cornerRadius: 12)
            } header: {
                sectionHeader("Analysis Result")
            }
            .transparentListRow()
        }
    }

    @ViewBuilder
    private var comparisonSection: some View {
        if shouldShowComparisonSection {
            Section {
                VStack(spacing: 16) {
                    #if os(iOS)
                    if isLivePhotoInput, let inputLP = inputLivePhoto, let outputLP = outputLivePhoto {
                        VStack(alignment: .leading, spacing: 5) {
                            Label("Input Live Photo", systemImage: "livephoto")
                                .font(AppTypography.font(size: 11))
                                .foregroundColor(.secondary)
                            ZStack(alignment: .bottomTrailing) {
                                LivePhotoPreviewView(
                                    livePhoto: inputLP,
                                    replayToken: livePhotoSyncReplayToken,
                                    autoPlay: false
                                )
                                .id("comparison-input-live-\(inputLivePhotoViewID)")
                                .frame(maxWidth: .infinity)
                                .frame(height: livePhotoPreviewHeight)
                                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                            }
                        }

                        Rectangle()
                            .fill(Color.white.opacity(0.28))
                            .frame(height: 1)

                        VStack(alignment: .leading, spacing: 5) {
                            Label("Output Live Photo", systemImage: "checkmark.circle.fill")
                                .font(AppTypography.font(size: 11))
                                .foregroundColor(.accentColor)
                            LivePhotoPreviewView(
                                livePhoto: outputLP,
                                replayToken: livePhotoSyncReplayToken,
                                autoPlay: false
                            )
                            .id("comparison-output-live-\(outputLivePhotoViewID)")
                            .frame(maxWidth: .infinity)
                            .frame(height: livePhotoPreviewHeight)
                            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        }

                        HStack {
                            Spacer()
                            Button {
                                playSynchronizedLivePhotoPreview()
                            } label: {
                                Label("Play", systemImage: "play.fill")
                                    .font(AppTypography.font(size: 14, weight: .semibold))
                                    .frame(width: 160, alignment: .center)
                                    .foregroundStyle(Color.accentColor)
                                    .glassButton(cornerRadius: 10)
                            }
                            .buttonStyle(.plain)
                            .opacity(isSyncLivePhotoPlaying ? 0.45 : 1.0)
                            .disabled(isSyncLivePhotoPlaying)
                            Spacer()
                        }

                        HStack {
                            Spacer()
                            Button(action: saveOutputImage) {
                                Text("Save")
                                    .font(AppTypography.font(size: 14, weight: .semibold))
                                    .frame(width: 140, alignment: .center)
                                    .foregroundStyle(Color.accentColor)
                                    .glassButton(cornerRadius: 10)
                            }
                            .buttonStyle(.plain)
                            .disabled(
                                outputLivePhotoPhotoURL == nil ||
                                outputLivePhotoVideoURL == nil ||
                                model.running
                            )
                            Spacer()
                        }
                    } else {
                        comparisonStillImagesContent
                    }
                    #else
                    comparisonStillImagesContent
                    #endif
                }
                .padding(.vertical, 8)
                .glassCard(cornerRadius: 12)
                .transition(.opacity.animation(.easeIn(duration: 0.5)))
            } header: {
                sectionHeader("Comparison (Before & After)")
            }
            .transparentListRow()
        }
    }

    private var comparisonStillImagesContent: some View {
        Group {
            VStack(alignment: .leading, spacing: 5) {
                Label("Input Image", systemImage: "arrow.up.circle")
                    .font(AppTypography.font(size: 11))
                    .foregroundColor(.secondary)
                ImageContainerView(platformImage: uploadedImage, cgImage: nil, minHeight: minImageHeight) {
                    Text("Original image not available")
                }
            }

            Rectangle()
                .fill(Color.white.opacity(0.28))
                .frame(height: 1)

            VStack(alignment: .leading, spacing: 5) {
                Label("Output Image", systemImage: "checkmark.circle.fill")
                    .font(AppTypography.font(size: 11))
                    .foregroundColor(.accentColor)
                ImageContainerView(platformImage: nil, cgImage: model.retouchedImage, minHeight: minImageHeight) {
                    Text("Error loading output").foregroundColor(.red)
                }
                HStack {
                    Spacer()
                    Button(action: saveOutputImage) {
                        Text("Save")
                            .font(AppTypography.font(size: 14, weight: .semibold))
                            .frame(width: 140, alignment: .center)
                            .foregroundStyle(Color.accentColor)
                            .glassButton(cornerRadius: 10)
                    }
                    .buttonStyle(.plain)
                    .disabled(model.retouchedImage == nil || model.running)
                    Spacer()
                }
            }
        }
    }

    @ViewBuilder
    private var autoAdjustmentsSection: some View {
        if selectedMode == "Auto" && hasAutoRunResult {
            Section {
                HStack(spacing: 18) {
                    autoMaskToggleButton(
                        title: "Light",
                        systemImage: "sun.max.fill",
                        isOn: $autoEnableLight
                    )
                    autoMaskToggleButton(
                        title: "Global Color",
                        systemImage: "globe.americas.fill",
                        isOn: $autoEnableGlobalColor
                    )
                    autoMaskToggleButton(
                        title: "Specific Color",
                        systemImage: "paintpalette.fill",
                        isOn: $autoEnableSpecificColor
                    )
                }
                .frame(maxWidth: .infinity)
                .glassCard(cornerRadius: 12)

                HStack {
                    Spacer()
                    Button {
                        rerenderAutoAdjustments()
                    } label: {
                        Text("Rerender")
                            .font(AppTypography.font(size: 14, weight: .semibold))
                            .frame(width: 150, alignment: .center)
                            .foregroundStyle(Color.accentColor)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(
                                RoundedRectangle(cornerRadius: 10, style: .continuous)
                                    .fill(Color.clear)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 10, style: .continuous)
                                    .stroke(Color.white.opacity(0.70), lineWidth: 1.25)
                            )
                    }
                    .buttonStyle(.plain)
                    .disabled((model.running && !isManualRerendering) || sourceCIImage == nil)
                    Spacer()
                }
                .padding(.top, 6)
            } header: {
                sectionHeader("Auto Adjustments")
            }
            .transparentListRow()
        }
    }

    private var listBottomAnchor: some View {
        Color.clear
            .frame(height: 0.1)
            .id("ListBottomAnchor")
            .transparentListRow(horizontal: 0, vertical: 0)
    }

    var body: some View {
        VStack(spacing: 0) {
            topGlassHeader

            ScrollViewReader { scrollProxy in
                contentList(scrollProxy: scrollProxy)
            }
        }
        .font(AppTypography.font(size: 15))
        .background(atmosphericBackground)
        .overlay {
            if showWelcomeOverlay {
                welcomeOverlay
                    .transition(.opacity)
                    .zIndex(2)
            }

            if showFakeProgress {
                ZStack {
                    Color.black.opacity(0.18)
                        .ignoresSafeArea()
                    VStack(spacing: 10) {
                        ProgressView(value: fakeProgressValue, total: 1.0)
                            .progressViewStyle(.linear)
                        Text("Rendering output image... \(Int(fakeProgressValue * 100))%")
                            .font(AppTypography.font(size: 12, weight: .semibold))
                            .foregroundColor(.primary)
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 14)
                    .frame(width: 280)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .shadow(color: .black.opacity(0.18), radius: 12, y: 4)
                }
                .transition(.opacity)
                .zIndex(3)
            }
        }
        .safeAreaInset(edge: .bottom) {
            if model.running && model.isStreamingOutput {
                HStack {
                    Button {
                        stopAnalysis()
                    } label: {
                        Text("Stop")
                            .frame(maxWidth: .infinity, minHeight: 44)
                            .font(AppTypography.font(size: 17, weight: .semibold))
                            .foregroundColor(.white)
                            .background(
                                RoundedRectangle(cornerRadius: 12, style: .continuous)
                                    .fill(Color.red)
                            )
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 16)
                .padding(.top, 6)
                .padding(.bottom, 6)
                .background(.ultraThinMaterial)
            }
        }
        .task {
            Task {
                await model.load()
            }
            await runWelcomeAnimation()
        }
        #if os(iOS)
        .sheet(isPresented: $showLivePhotoPickerSheet) {
            LivePhotoPickerView { localId in
                guard let localId else {
                    Task { @MainActor in
                        await showSaveResult("Failed to resolve Live Photo asset identifier.")
                    }
                    return
                }
                Task {
                    let assets = PHAsset.fetchAssets(withLocalIdentifiers: [localId], options: nil)
                    guard let asset = assets.firstObject, asset.mediaSubtypes.contains(.photoLive) else {
                        await showSaveResult("Selected item is not a Live Photo.")
                        return
                    }
                    dismissKeyboard()
                    clearInferenceOutputsOnly()
                    await prepareLivePhotoInput(from: asset)
                }
            }
        }
        .ignoresSafeArea(.keyboard, edges: .bottom)
        #endif
        .preferredColorScheme(.light)
        .alert("Save Output Image", isPresented: $showSaveAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(saveAlertMessage)
        }
    }

    @ViewBuilder
    private var welcomeOverlay: some View {
        ZStack {
            Color.white
                .ignoresSafeArea()
            (
                Text("Welcom to ")
                    .foregroundStyle(Color.black.opacity(0.85))
                + Text("VeraRetouch")
                    .foregroundStyle(
                        LinearGradient(
                            colors: [
                                Color(red: 0.12, green: 0.47, blue: 0.98),
                                Color(red: 0.15, green: 0.78, blue: 0.57),
                                Color(red: 0.98, green: 0.57, blue: 0.18),
                            ],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                + Text(", my dear friend")
                    .foregroundStyle(Color.black.opacity(0.85))
            )
            .font(AppTypography.font(size: 25, weight: .semibold))
            .multilineTextAlignment(.center)
            .padding(.horizontal, 20)
            .lineLimit(2)
            .minimumScaleFactor(0.75)
            .opacity(welcomeTextVisible ? 1 : 0)
            .scaleEffect(welcomeTextVisible ? 1.0 : 0.97)
            .offset(y: welcomeFloatUp ? -8 : 8)
            .animation(.easeInOut(duration: 1.6).repeatForever(autoreverses: true), value: welcomeFloatUp)
        }
    }

    @MainActor
    private func runWelcomeAnimation() async {
        withAnimation(.easeOut(duration: 0.35)) {
            showWelcomeOverlay = true
        }
        withAnimation(.easeOut(duration: 0.45)) {
            welcomeTextVisible = false
        }
        try? await Task.sleep(nanoseconds: 220_000_000)
        withAnimation(.easeOut(duration: 0.55)) {
            welcomeTextVisible = true
        }
        welcomeFloatUp = true
        try? await Task.sleep(nanoseconds: 2_200_000_000)
        withAnimation(.easeIn(duration: 0.35)) {
            welcomeTextVisible = false
            showWelcomeOverlay = false
        }
    }

    func clearModeState() {
        model.output = ""
        model.retouchedImage = nil
        model.promptTime = ""
        model.retouchPipelineTime = ""
        model.retouchTimingBreakdown = []

        resetDictionary(dict: &lightParams)
        resetDictionary(dict: &tempParams)
        resetDictionary(dict: &mixerParams)
        styleInstruction = Self.defaultStyleInstruction
        styleInstructionIsDefault = true
        selectedStyleTag = "Dreamy Pink"
        autoRunRequested = false
        hasAutoRunResult = false
        // Keep uploaded media (including Live Photo) when switching modes.
        // Only inference outputs and mode-specific controls are reset here.
        #if os(iOS)
        outputLivePhoto = nil
        outputLivePhotoPhotoURL = nil
        outputLivePhotoVideoURL = nil
        livePhotoSyncReplayToken = 0
        isInputLivePhotoPlaying = false
        isSyncLivePhotoPlaying = false
        #endif
    }

    @ViewBuilder
    func autoMaskToggleButton(title: String, systemImage: String, isOn: Binding<Bool>) -> some View {
        Button {
            isOn.wrappedValue.toggle()
        } label: {
            VStack(spacing: 6) {
                Image(systemName: systemImage)
                    .font(AppTypography.font(size: 18, weight: .semibold))
                    .foregroundStyle(isOn.wrappedValue ? Color.accentColor : Color.secondary.opacity(0.75))
                    .frame(width: 42, height: 42)
                    .background(
                        Circle()
                            .fill(.ultraThinMaterial)
                            .overlay(
                                Circle()
                                    .stroke(
                                        isOn.wrappedValue ? Color.white.opacity(0.82) : Color.white.opacity(0.60),
                                        lineWidth: 1.1
                                    )
                            )
                    )
                Text(title)
                    .font(AppTypography.font(size: 11, weight: .medium))
                    .foregroundStyle(isOn.wrappedValue ? Color.accentColor : Color.secondary.opacity(0.78))
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
                    .frame(maxWidth: 90)
            }
            .opacity(isOn.wrappedValue ? 1.0 : 0.9)
        }
        .buttonStyle(.plain)
        .disabled(model.running)
    }

    func autoRetouchMask() -> [Float32] {
        [
            autoEnableLight ? 1.0 : 0.0,
            autoEnableGlobalColor ? 1.0 : 0.0,
            autoEnableSpecificColor ? 1.0 : 0.0,
        ]
    }

    func stylePromptText(for tag: String) -> String {
        "I want a \(tag) style."
    }

    func applyStylePresetTag(_ tag: String) {
        selectedStyleTag = tag
        styleInstruction = stylePromptText(for: tag)
        styleInstructionIsDefault = false
    }

    func isAllZeroMask(_ mask: [Float32]) -> Bool {
        mask.allSatisfy { abs($0) < 0.0001 }
    }

    func handleAutoMaskToggleChanged() {
        guard selectedMode == "Auto" else { return }
        guard hasAutoRunResult else { return }
        guard !model.running else { return }
        guard sourceCIImage != nil else { return }
        rerenderAutoAdjustments()
    }

    func rerenderAutoAdjustments() {
        let mask = autoRetouchMask()
        if isAllZeroMask(mask) {
            stopFakeProgressImmediately()
            if isLivePhotoInput {
                #if os(iOS)
                outputLivePhoto = inputLivePhoto
                outputLivePhotoPhotoURL = nil
                outputLivePhotoVideoURL = nil
                #endif
            } else if let ci = sourceCIImage {
                let extent = ci.extent.integral
                let context = CIContext(options: nil)
                model.retouchedImage = context.createCGImage(ci, from: extent)
            }
            model.output =
                (model.output.isEmpty ? "" : model.output + "\n\n")
                + "No adjustments enabled. Showing the input preview."
            hasAutoRunResult = true
            return
        }

        Task {
            await MainActor.run {
                isManualRerendering = true
                startFakeProgress()
            }
            defer {
                Task { @MainActor in
                    isManualRerendering = false
                    completeFakeProgress()
                }
            }

            let lutDimension: Int? = {
                switch selectedLUTMode {
                case "33": return 33
                case "65": return 65
                default: return nil
                }
            }()

            #if os(iOS)
            if isLivePhotoInput {
                let ok = await model.rerenderFromCachedRetouch(
                    retouchMask: mask,
                    lutDimension: lutDimension
                )
                guard
                    ok,
                    let cubeData = model.lastLUTCubeData,
                    let cubeDim = model.lastLUTDimension,
                    let asset = livePhotoAsset,
                    let inputVideoURL = livePhotoVideoURL
                else {
                    autoFollowOutput = true
                    runAnalysis()
                    return
                }
                do {
                    let (processedPhotoURL, processedVideoURL) = try await renderLivePhotoWithLUT(
                        asset: asset,
                        inputVideoURL: inputVideoURL,
                        cubeData: cubeData,
                        cubeDimension: cubeDim
                    )
                    let built = await Self.buildLivePhoto(photoURL: processedPhotoURL, videoURL: processedVideoURL)
                    await MainActor.run {
                        outputLivePhoto = built
                        outputLivePhotoViewID &+= 1
                        outputLivePhotoPhotoURL = processedPhotoURL
                        outputLivePhotoVideoURL = processedVideoURL
                    }
                } catch {
                    await showSaveResult("Live Photo rerender failed: \(error.localizedDescription)")
                }
                return
            }
            #endif

            let ok = await model.rerenderFromCachedRetouch(
                retouchMask: mask,
                lutDimension: lutDimension
            )
            if !ok {
                // Fallback to full run if cache is unavailable.
                autoFollowOutput = true
                runAnalysis()
            }
        }
    }

    func clearInferenceOutputsOnly() {
        model.output = ""
        model.retouchedImage = nil
        model.promptTime = ""
        model.retouchPipelineTime = ""
        model.retouchTimingBreakdown = []
        autoRunRequested = false
        hasAutoRunResult = false
        #if os(iOS)
        outputLivePhoto = nil
        outputLivePhotoPhotoURL = nil
        outputLivePhotoVideoURL = nil
        outputLivePhotoViewID = 0
        livePhotoSyncReplayToken = 0
        isSyncLivePhotoPlaying = false
        #endif
    }

    func resetDictionary(dict: inout [String: Double]) {
        for key in dict.keys {
            dict[key] = 0.0
        }
    }

    func resetAllProfessionalParams() {
        resetDictionary(dict: &lightParams)
        resetDictionary(dict: &tempParams)
        resetDictionary(dict: &mixerParams)
    }

    #if os(iOS)
    func dismissKeyboard() {
        styleEditorFocused = false
        UIApplication.shared.sendAction(
            #selector(UIResponder.resignFirstResponder),
            to: nil,
            from: nil,
            for: nil
        )
    }

    #endif

    func stopAnalysis() {
        autoFollowOutput = false
        autoRunRequested = false
        isManualRerendering = false
        stopFakeProgressImmediately()
        model.stopGeneration()
        let stopMessage = "Generation stopped by user."
        if model.output.isEmpty {
            model.output = stopMessage
        } else if !model.output.contains(stopMessage) {
            model.output += "\n\n" + stopMessage
        }
    }

    func runAnalysis() {
        guard let ciImage = sourceCIImage else { return }

        stopFakeProgressImmediately()
        isManualRerendering = false
        sawOutputInCurrentRun = false
        lastOutputChangeAt = nil

        model.output = ""
        model.retouchedImage = nil
        model.promptTime = ""
        model.retouchPipelineTime = ""
        model.retouchTimingBreakdown = []

        if selectedMode == "Reference" {
            runReferenceRetouch(sourceImage: ciImage)
            return
        }

        let prompt = buildPromptForCurrentMode()
        let userInput = UserInput(
            prompt: .text(prompt),
            images: [.ciImage(ciImage)]
        )
        let retouchMask: [Float32]
        if selectedMode == "Professional" {
            retouchMask = professionalRetouchMask()
        } else if selectedMode == "Auto" {
            retouchMask = autoRetouchMask()
            autoRunRequested = true
        } else {
            retouchMask = [1.0, 1.0, 1.0]
        }

        Task {
            let lutDimension: Int?
            switch selectedLUTMode {
            case "33":
                lutDimension = 33
            case "65":
                lutDimension = 65
            default:
                lutDimension = nil
            }

            let effectiveTemperature: Double = (selectedMode == "Professional") ? 0.2 : samplingTemperature
            let effectiveTopP: Double = (selectedMode == "Professional") ? 0.8 : samplingTopP
            #if os(iOS)
            if isLivePhotoInput {
                await runLivePhotoPipeline(
                    prompt: prompt,
                    retouchMask: retouchMask,
                    lutDimension: lutDimension ?? 65,
                    temperature: Float(effectiveTemperature),
                    topP: Float(effectiveTopP)
                )
            } else {
                await model.generate(
                    userInput,
                    retouchMask: retouchMask,
                    lutDimension: lutDimension,
                    temperature: Float(effectiveTemperature),
                    topP: Float(effectiveTopP)
                )
            }
            #else
            await model.generate(
                userInput,
                retouchMask: retouchMask,
                lutDimension: lutDimension,
                temperature: Float(effectiveTemperature),
                topP: Float(effectiveTopP)
            )
            #endif
        }
    }

    func runReferenceRetouch(sourceImage: CIImage) {
        guard let referenceBeforeCIImage, let referenceAfterCIImage else { return }

        Task {
            let lutDimension: Int?
            switch selectedLUTMode {
            case "33":
                lutDimension = 33
            case "65":
                lutDimension = 65
            default:
                lutDimension = nil
            }

            await MainActor.run {
                startFakeProgress()
            }
            defer {
                Task { @MainActor in
                    completeFakeProgress()
                }
            }

            _ = await model.referenceRetouch(
                sourceImage: sourceImage,
                referenceBeforeImage: referenceBeforeCIImage,
                referenceAfterImage: referenceAfterCIImage,
                lutDimension: lutDimension
            )
        }
    }

    func applyAutoRendererModeIfNeeded(for image: CIImage) {
        // Keep manual selection unless current mode is Off.
        if selectedLUTMode != "Off" {
            return
        }
        let extent = image.extent.integral
        let width = max(0, Int(extent.width))
        let height = max(0, Int(extent.height))
        let pixelCount = width * height
        if pixelCount > autoLUTPixelThreshold {
            selectedLUTMode = "33"
        } else {
            selectedLUTMode = "33"
        }
    }

    func buildPromptForCurrentMode() -> String {
        switch selectedMode {
        case "Auto":
            return Self.autoRetouchPrompt
        case "Style":
            let instruction = styleInstruction.trimmingCharacters(in: .whitespacesAndNewlines)
            if instruction.isEmpty {
                return Self.styleRetouchPrompt
            }
            return Self.styleRetouchPrompt + "\n" + instruction
        case "Professional":
            return Self.professionalRetouchPrompt + " " + professionalParamsInstructionText()
        default:
            return Self.autoRetouchPrompt
        }
    }

    var fixedPromptForCurrentMode: String {
        switch selectedMode {
        case "Auto":
            return Self.autoRetouchPrompt
        case "Style":
            return Self.styleRetouchPrompt
        case "Professional":
            return Self.professionalRetouchPrompt
        default:
            return Self.autoRetouchPrompt
        }
    }

    var runButtonTitle: String {
        selectedMode == "Reference" ? "Run Reference Retouch" : "Run"
    }

    var canRunCurrentMode: Bool {
        guard sourceCIImage != nil, !model.running else { return false }
        guard selectedMode == "Reference" else { return true }
        return referenceBeforeCIImage != nil && referenceAfterCIImage != nil
    }

    func normalizedProfessionalGroups() -> ([String: Double], [String: Double], [String: Double]) {
        func normalized(_ raw: Double) -> Double {
            (raw / 100.0 * 1000.0).rounded() / 1000.0
        }

        var light: [String: Double] = [:]
        var temp: [String: Double] = [:]
        var mixer: [String: Double] = [:]

        for key in lightKeys {
            light[key] = normalized(lightParams[key] ?? 0.0)
        }
        for key in tempKeys {
            temp[key] = normalized(tempParams[key] ?? 0.0)
        }
        for key in mixerKeys {
            mixer[key] = normalized(mixerParams[key] ?? 0.0)
        }
        return (light, temp, mixer)
    }

    func allZero(_ data: [String: Double]) -> Bool {
        for value in data.values {
            if value > 0.0001 || value < -0.0001 {
                return false
            }
        }
        return true
    }

    func professionalRetouchMask() -> [Float32] {
        let (light, temp, mixer) = normalizedProfessionalGroups()
        let mask1: Float32 = allZero(light) ? 0.0 : 1.0
        let mask2: Float32 = allZero(temp) ? 0.0 : 1.0
        let mask3: Float32 = allZero(mixer) ? 0.0 : 1.0
        return [mask1, mask2, mask3]
    }

    func dictionaryText(_ title: String, keys: [String], values: [String: Double]) -> String {
        let body = keys.map { key in
            let v = values[key] ?? 0.0
            return "\"\(key)\": \(String(format: "%.3f", v))"
        }.joined(separator: ", ")
        return "\"\(title)\": { \(body) }"
    }

    func professionalParamsInstructionText() -> String {
        let (light, temp, mixer) = normalizedProfessionalGroups()
        let blocks = [
            dictionaryText("Light Adjustment", keys: lightKeys, values: light),
            dictionaryText("Color and Temperature Adjustment", keys: tempKeys, values: temp),
            dictionaryText("Specific Color Adjustment", keys: mixerKeys, values: mixer),
        ]
        return "{ " + blocks.joined(separator: ", ") + " }"
    }

    #if os(iOS)
    func prepareLivePhotoInput(from asset: PHAsset) async {
        var auth = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        if auth == .notDetermined {
            auth = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        }
        guard auth == .authorized || auth == .limited else {
            await showSaveResult("Photo Library access denied. Please allow access in Settings.")
            return
        }

        isLivePhotoInput = true
        livePhotoAsset = asset

        // Clear previous still-image outputs immediately to avoid stale layout artifacts
        // when switching from image run -> live photo run.
        model.output = ""
        model.retouchedImage = nil
        model.promptTime = ""
        model.retouchPipelineTime = ""
        model.retouchTimingBreakdown = []

        selectedLUTMode = "33"  // Prefer 33x33x33 on iPhone for better stability/performance.
        outputLivePhoto = nil
        outputLivePhotoPhotoURL = nil
        outputLivePhotoVideoURL = nil
        outputLivePhotoViewID = 0
        livePhotoDurationSeconds = 0
        livePhotoFrameProgress = 0.5
        livePhotoFrameRequestID = 0
        livePhotoReferenceFrameImage = nil
        livePhotoSyncReplayToken = 0
        isInputLivePhotoPlaying = false
        isSyncLivePhotoPlaying = false
        livePhotoPreviewTask?.cancel()
        livePhotoPreviewTask = nil

        do {
            if let stillImage = try await requestUIImage(for: asset) {
                let normalized = stillImage.normalizedUp()
                uploadedImage = normalized
                sourceCIImage = CIImage(image: normalized)
            }

            inputLivePhoto = try await requestLivePhoto(for: asset)
            inputLivePhotoViewID &+= 1

            let resources = PHAssetResource.assetResources(for: asset)
            guard let photoRes = resources.first(where: { $0.type == .photo || $0.type == .fullSizePhoto }) else {
                await showSaveResult("Live Photo still resource not found.")
                return
            }
            guard let pairedVideoResource = resources.first(where: { $0.type == .pairedVideo }) else {
                await showSaveResult("Live Photo paired video resource not found.")
                return
            }
            let photoURL = try await writeAssetResourceToTemporaryFile(resource: photoRes)
            let videoURL = try await writeAssetResourceToTemporaryFile(resource: pairedVideoResource, fileExtension: "mov")
            livePhotoVideoURL = videoURL

            livePhotoDurationSeconds = try await Self.videoDurationSeconds(from: videoURL)
            if livePhotoDurationSeconds > 0 {
                if let ref = try await Self.frameUIImage(from: videoURL, progress: livePhotoFrameProgress) {
                    let normalized = ref.normalizedUp()
                    uploadedImage = normalized
                    sourceCIImage = CIImage(image: normalized)
                    livePhotoReferenceFrameImage = normalized
                }
            }
        } catch {
            await showSaveResult("Failed to prepare Live Photo: \(error.localizedDescription)")
        }
    }

    func livePhotoPlaybackDuration() -> Double {
        if livePhotoDurationSeconds.isFinite, livePhotoDurationSeconds > 0 {
            return min(max(livePhotoDurationSeconds, 0.8), 15.0)
        }
        return 2.0
    }

    func playInputLivePhotoPreview() {
        guard !isInputLivePhotoPlaying else { return }
        isInputLivePhotoPlaying = true
        inputLivePhotoReplayToken &+= 1
        let seconds = livePhotoPlaybackDuration()
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            isInputLivePhotoPlaying = false
        }
    }

    func playSynchronizedLivePhotoPreview() {
        guard !isSyncLivePhotoPlaying else { return }
        isSyncLivePhotoPlaying = true
        livePhotoSyncReplayToken &+= 1
        let seconds = livePhotoPlaybackDuration()
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            isSyncLivePhotoPlaying = false
        }
    }

    func updateLivePhotoReferenceFrame(progress: Double) {
        guard isLivePhotoInput, let videoURL = livePhotoVideoURL else { return }
        let clamped = max(0, min(1, progress))
        livePhotoPreviewTask?.cancel()
        livePhotoFrameRequestID &+= 1
        let requestID = livePhotoFrameRequestID
        livePhotoPreviewTask = Task {
            guard let frame = try? await Self.frameUIImage(from: videoURL, progress: clamped), !Task.isCancelled else {
                return
            }
            let normalized = frame.normalizedUp()
            await MainActor.run {
                guard requestID == livePhotoFrameRequestID else { return }
                uploadedImage = normalized
                sourceCIImage = CIImage(image: normalized)
                livePhotoReferenceFrameImage = normalized
            }
        }
    }

    func formattedTimestamp(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "00:00.00" }
        let minutes = Int(seconds) / 60
        let secs = seconds.truncatingRemainder(dividingBy: 60)
        return String(format: "%02d:%05.2f", minutes, secs)
    }

    func runLivePhotoPipeline(
        prompt: String,
        retouchMask: [Float32],
        lutDimension: Int,
        temperature: Float,
        topP: Float
    ) async {
        guard let asset = livePhotoAsset else {
            await showSaveResult("Live Photo asset is missing.")
            return
        }
        guard let videoURL = livePhotoVideoURL else {
            await showSaveResult("Live Photo video is missing.")
            return
        }
        guard livePhotoDurationSeconds > 0 else {
            await showSaveResult("Live Photo duration metadata is missing.")
            return
        }

        do {
            let progress = max(0, min(1, livePhotoFrameProgress))
            let refSeconds = livePhotoDurationSeconds * progress
            let refTime = CMTime(seconds: refSeconds, preferredTimescale: 600)
            let referenceCI = try referenceFrameCIImage(videoURL: videoURL, at: refTime)
            let userInput = UserInput(prompt: .text(prompt), images: [.ciImage(referenceCI)])

            await model.generateAndWait(
                userInput,
                retouchMask: retouchMask,
                lutDimension: lutDimension,
                temperature: temperature,
                topP: topP
            )

            guard let cubeData = model.lastLUTCubeData, let cubeDim = model.lastLUTDimension else {
                await showSaveResult("Failed to produce LUT from reference frame.")
                return
            }

            let (processedPhotoURL, processedVideoURL) = try await renderLivePhotoWithLUT(
                asset: asset,
                inputVideoURL: videoURL,
                cubeData: cubeData,
                cubeDimension: cubeDim
            )
            let builtLivePhoto = await Self.buildLivePhoto(photoURL: processedPhotoURL, videoURL: processedVideoURL)
            await MainActor.run {
                outputLivePhoto = builtLivePhoto
                outputLivePhotoViewID &+= 1
                outputLivePhotoPhotoURL = processedPhotoURL
                outputLivePhotoVideoURL = processedVideoURL
            }
        } catch {
            await showSaveResult("Live Photo processing failed: \(error.localizedDescription)")
        }
    }

    func requestUIImage(for asset: PHAsset) async throws -> UIImage? {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<UIImage?, Error>) in
            let options = PHImageRequestOptions()
            options.isNetworkAccessAllowed = true
            options.deliveryMode = .highQualityFormat
            options.resizeMode = .none
            options.isSynchronous = false

            PHImageManager.default().requestImageDataAndOrientation(for: asset, options: options) {
                data, _, _, info in
                if let cancelled = info?[PHImageCancelledKey] as? Bool, cancelled {
                    continuation.resume(returning: nil)
                    return
                }
                if let err = info?[PHImageErrorKey] as? Error {
                    continuation.resume(throwing: err)
                    return
                }
                if let data, let image = UIImage(data: data) {
                    continuation.resume(returning: image)
                } else {
                    continuation.resume(returning: nil)
                }
            }
        }
    }

    func requestLivePhoto(for asset: PHAsset) async throws -> PHLivePhoto? {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<PHLivePhoto?, Error>) in
            let options = PHLivePhotoRequestOptions()
            options.isNetworkAccessAllowed = true
            options.deliveryMode = .highQualityFormat
            PHImageManager.default().requestLivePhoto(
                for: asset,
                targetSize: PHImageManagerMaximumSize,
                contentMode: .aspectFit,
                options: options
            ) { livePhoto, info in
                if let cancelled = info?[PHImageCancelledKey] as? Bool, cancelled {
                    continuation.resume(returning: nil)
                    return
                }
                if let err = info?[PHImageErrorKey] as? Error {
                    continuation.resume(throwing: err)
                    return
                }
                continuation.resume(returning: livePhoto)
            }
        }
    }

    func writeAssetResourceToTemporaryFile(resource: PHAssetResource, fileExtension: String? = nil) async throws -> URL {
        let ext: String = {
            if let fileExtension, !fileExtension.isEmpty { return fileExtension }
            let originalExt = (resource.originalFilename as NSString).pathExtension
            if !originalExt.isEmpty { return originalExt.lowercased() }
            return "dat"
        }()
        let tmpURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension(ext)

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            PHAssetResourceManager.default().writeData(for: resource, toFile: tmpURL, options: nil) { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: ())
                }
            }
        }
        return tmpURL
    }

    static func videoDurationSeconds(from videoURL: URL) async throws -> Double {
        let asset = AVAsset(url: videoURL)
        let duration = try await asset.load(.duration)
        let durationSeconds = CMTimeGetSeconds(duration)
        guard durationSeconds.isFinite, durationSeconds > 0 else { return 0 }
        return durationSeconds
    }

    static func frameUIImage(from videoURL: URL, progress: Double) async throws -> UIImage? {
        let asset = AVAsset(url: videoURL)
        let duration = try await asset.load(.duration)
        let durationSeconds = CMTimeGetSeconds(duration)
        guard durationSeconds.isFinite, durationSeconds > 0 else { return nil }

        let clamped = max(0, min(1, progress))
        let t = CMTime(seconds: durationSeconds * clamped, preferredTimescale: 600)

        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.requestedTimeToleranceAfter = CMTime(seconds: 0.02, preferredTimescale: 600)
        gen.requestedTimeToleranceBefore = CMTime(seconds: 0.02, preferredTimescale: 600)
        gen.maximumSize = CGSize(width: 1024, height: 1024)
        if let cg = try? gen.copyCGImage(at: t, actualTime: nil) {
            return UIImage(cgImage: cg)
        }
        return nil
    }

    static func buildLivePhoto(photoURL: URL, videoURL: URL) async -> PHLivePhoto? {
        await withCheckedContinuation { (continuation: CheckedContinuation<PHLivePhoto?, Never>) in
            var didResume = false
            PHLivePhoto.request(
                withResourceFileURLs: [photoURL, videoURL],
                placeholderImage: nil,
                targetSize: PHImageManagerMaximumSize,
                contentMode: .aspectFit
            ) { livePhoto, info in
                if didResume { return }
                let degraded = (info[PHLivePhotoInfoIsDegradedKey] as? Bool) ?? false
                if degraded { return }
                didResume = true
                continuation.resume(returning: livePhoto)
            }
        }
    }

    func referenceFrameCIImage(videoURL: URL, at time: CMTime) throws -> CIImage {
        let asset = AVAsset(url: videoURL)
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.requestedTimeToleranceAfter = .zero
        gen.requestedTimeToleranceBefore = .zero
        let cg = try gen.copyCGImage(at: time, actualTime: nil)
        guard let ci = CIImage(image: UIImage(cgImage: cg).normalizedUp()) else {
            throw NSError(domain: "LivePhoto", code: -11, userInfo: [NSLocalizedDescriptionKey: "Failed to build reference frame image."])
        }
        return ci
    }

    func renderLivePhotoWithLUT(
        asset: PHAsset,
        inputVideoURL: URL,
        cubeData: Data,
        cubeDimension: Int
    ) async throws -> (photoURL: URL, videoURL: URL) {
        let resources = PHAssetResource.assetResources(for: asset)
        guard let photoRes = resources.first(where: { $0.type == .photo || $0.type == .fullSizePhoto }) else {
            throw NSError(domain: "LivePhoto", code: -21, userInfo: [NSLocalizedDescriptionKey: "Missing Live Photo still image resource."])
        }
        let originalPhotoURL = try await writeAssetResourceToTemporaryFile(resource: photoRes, fileExtension: "jpg")

        let processedPhotoURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("livephoto_processed_\(UUID().uuidString).jpg")
        let processedVideoURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("livephoto_processed_\(UUID().uuidString).mov")

        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        guard let colorSpace else {
            throw NSError(domain: "LivePhoto", code: -30, userInfo: [NSLocalizedDescriptionKey: "sRGB color space unavailable."])
        }
        let ciContext = CIContext(options: nil)

        if let photoCI = CIImage(contentsOf: originalPhotoURL),
            let filteredPhoto = applyLUTCubeData(cubeData, dimension: cubeDimension, to: photoCI)
        {
            let rep = ciContext.jpegRepresentation(of: filteredPhoto, colorSpace: colorSpace, options: [:])
            if let rep {
                try rep.write(to: processedPhotoURL, options: .atomic)
            } else {
                throw NSError(domain: "LivePhoto", code: -31, userInfo: [NSLocalizedDescriptionKey: "Failed to encode processed Live Photo still."])
            }
        } else {
            throw NSError(domain: "LivePhoto", code: -32, userInfo: [NSLocalizedDescriptionKey: "Failed to filter Live Photo still image."])
        }

        let videoAsset = AVAsset(url: inputVideoURL)
        guard let export = AVAssetExportSession(asset: videoAsset, presetName: AVAssetExportPresetHighestQuality) else {
            throw NSError(domain: "LivePhoto", code: -33, userInfo: [NSLocalizedDescriptionKey: "Unable to create AVAssetExportSession."])
        }
        let composition = AVVideoComposition(asset: videoAsset) { request in
            let src = request.sourceImage.clampedToExtent()
            if let out = applyLUTCubeData(cubeData, dimension: cubeDimension, to: src) {
                request.finish(with: out.cropped(to: request.sourceImage.extent), context: nil)
            } else {
                request.finish(with: request.sourceImage, context: nil)
            }
        }
        export.videoComposition = composition
        export.outputURL = processedVideoURL
        export.outputFileType = .mov
        export.shouldOptimizeForNetworkUse = false
        try? FileManager.default.removeItem(at: processedVideoURL)

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            export.exportAsynchronously {
                switch export.status {
                case .completed:
                    continuation.resume(returning: ())
                case .failed:
                    continuation.resume(throwing: export.error ?? NSError(domain: "LivePhoto", code: -34))
                case .cancelled:
                    continuation.resume(throwing: NSError(domain: "LivePhoto", code: -35, userInfo: [NSLocalizedDescriptionKey: "Video export cancelled."]))
                default:
                    continuation.resume(throwing: NSError(domain: "LivePhoto", code: -36, userInfo: [NSLocalizedDescriptionKey: "Video export did not complete."]))
                }
            }
        }

        return (processedPhotoURL, processedVideoURL)
    }

    func applyLUTCubeData(_ cubeData: Data, dimension: Int, to image: CIImage) -> CIImage? {
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        if let filter = CIFilter(name: "CIColorCubeWithColorSpace") {
            filter.setValue(image, forKey: kCIInputImageKey)
            filter.setValue(dimension, forKey: "inputCubeDimension")
            filter.setValue(cubeData, forKey: "inputCubeData")
            filter.setValue(colorSpace, forKey: "inputColorSpace")
            return filter.outputImage
        } else if let filter = CIFilter(name: "CIColorCube") {
            filter.setValue(image, forKey: kCIInputImageKey)
            filter.setValue(dimension, forKey: "inputCubeDimension")
            filter.setValue(cubeData, forKey: "inputCubeData")
            return filter.outputImage
        }
        return nil
    }

    func saveProcessedLivePhoto(photoURL: URL, pairedVideoURL: URL) async throws {
        try await PHPhotoLibrary.shared().performChanges {
            let request = PHAssetCreationRequest.forAsset()
            request.addResource(with: .photo, fileURL: photoURL, options: nil)
            request.addResource(with: .pairedVideo, fileURL: pairedVideoURL, options: nil)
        }
    }
    #endif

    func startFakeProgress() {
        fakeProgressTask?.cancel()
        withAnimation(.easeOut(duration: 0.12)) {
            fakeProgressValue = 0.0
        }
        showFakeProgress = true
        let start = Date()
        fakeProgressTask = Task {
            while !Task.isCancelled {
                let elapsed = Date().timeIntervalSince(start)
                let target: Double
                if elapsed < 2.4 {
                    target = 0.60 * (elapsed / 2.4)
                } else if elapsed < 6.0 {
                    target = 0.60 + 0.30 * ((elapsed - 2.4) / 3.6)
                } else {
                    let t = elapsed - 6.0
                    target = 0.90 + 0.09 * (1 - exp(-t / 18.0))
                }
                let clamped = min(0.99, max(0.0, target))
                await MainActor.run {
                    if clamped > fakeProgressValue {
                        withAnimation(.linear(duration: 0.10)) {
                            fakeProgressValue = clamped
                        }
                    }
                }
                try? await Task.sleep(nanoseconds: 33_000_000)
            }
        }
    }

    func stopFakeProgressImmediately() {
        fakeProgressTask?.cancel()
        fakeProgressTask = nil
        showFakeProgress = false
        fakeProgressValue = 0
    }

    func completeFakeProgress() {
        guard showFakeProgress else { return }
        fakeProgressTask?.cancel()
        fakeProgressTask = nil
        Task { @MainActor in
            withAnimation(.easeOut(duration: 0.22)) {
                fakeProgressValue = 1.0
            }
            try? await Task.sleep(nanoseconds: 380_000_000)
            withAnimation(.easeOut(duration: 0.18)) {
                showFakeProgress = false
            }
            try? await Task.sleep(nanoseconds: 180_000_000)
            fakeProgressValue = 0
        }
    }

    @MainActor
    func setReferenceImage(_ image: PlatformImage, ciImage: CIImage?, kind: String) {
        clearInferenceOutputsOnly()
        if kind == "before" {
            referenceBeforeImage = image
            referenceBeforeCIImage = ciImage
        } else {
            referenceAfterImage = image
            referenceAfterCIImage = ciImage
        }
    }

    func loadReferencePickerItem(_ item: PhotosPickerItem?, kind: String) async {
        guard let item else { return }
        guard let data = try? await item.loadTransferable(type: Data.self) else { return }
        #if os(iOS)
        guard let uiImage = UIImage(data: data) else { return }
        let normalized = uiImage.normalizedUp()
        await setReferenceImage(normalized, ciImage: CIImage(image: normalized), kind: kind)
        #elseif os(macOS)
        guard let nsImage = NSImage(data: data) else { return }
        await setReferenceImage(nsImage, ciImage: CIImage(data: data), kind: kind)
        #endif
    }

    #if os(macOS)
    func selectReferenceImageMac(kind: String) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.image]
        if panel.runModal() == .OK,
            let url = panel.url,
            let nsImage = NSImage(contentsOf: url)
        {
            setReferenceImage(nsImage, ciImage: CIImage(contentsOf: url), kind: kind)
        }
    }

    func selectImageMac() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.image]
        if panel.runModal() == .OK,
            let url = panel.url,
            let nsImage = NSImage(contentsOf: url)
        {
            uploadedImage = nsImage
            sourceCIImage = CIImage(contentsOf: url)
            isLivePhotoInput = false
            selectedLUTMode = "Off"
            if let image = sourceCIImage {
                applyAutoRendererModeIfNeeded(for: image)
            }
        }
    }
    #endif

    @MainActor
    func showSaveResult(_ message: String) {
        saveAlertMessage = message
        showSaveAlert = true
    }

    func saveOutputImage() {
        #if os(iOS)
        if isLivePhotoInput, let photoURL = outputLivePhotoPhotoURL, let videoURL = outputLivePhotoVideoURL {
            Task {
                await saveOutputLivePhotoToPhotoLibrary(photoURL: photoURL, pairedVideoURL: videoURL)
            }
            return
        }
        #endif

        guard let cgImage = model.retouchedImage else {
            showSaveResult("No output image to save.")
            return
        }

        #if os(iOS)
        Task {
            await saveOutputImageToPhotoLibrary(cgImage: cgImage)
        }
        #elseif os(macOS)
        saveOutputImageOnMac(cgImage: cgImage)
        #endif
    }

    #if os(iOS)
    func saveOutputImageToPhotoLibrary(cgImage: CGImage) async {
        let uiImage = UIImage(cgImage: cgImage)
        guard let pngData = uiImage.pngData() else {
            await showSaveResult("Failed to encode image.")
            return
        }

        var authStatus = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        if authStatus == .notDetermined {
            authStatus = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        }

        guard authStatus == .authorized || authStatus == .limited else {
            await showSaveResult("Photo Library permission denied.")
            return
        }

        do {
            try await PHPhotoLibrary.shared().performChanges {
                let request = PHAssetCreationRequest.forAsset()
                request.addResource(with: .photo, data: pngData, options: nil)
            }
            await showSaveResult("Saved to Photos.")
        } catch {
            await showSaveResult("Save failed: \(error.localizedDescription)")
        }
    }

    func saveOutputLivePhotoToPhotoLibrary(photoURL: URL, pairedVideoURL: URL) async {
        var authStatus = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        if authStatus == .notDetermined {
            authStatus = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        }

        guard authStatus == .authorized || authStatus == .limited else {
            await showSaveResult("Photo Library permission denied.")
            return
        }

        do {
            try await PHPhotoLibrary.shared().performChanges {
                let request = PHAssetCreationRequest.forAsset()
                request.addResource(with: .photo, fileURL: photoURL, options: nil)
                request.addResource(with: .pairedVideo, fileURL: pairedVideoURL, options: nil)
            }
            await showSaveResult("Live Photo saved to Photos.")
        } catch {
            await showSaveResult("Live Photo save failed: \(error.localizedDescription)")
        }
    }
    #endif

    #if os(macOS)
    func saveOutputImageOnMac(cgImage: CGImage) {
        let panel = NSSavePanel()
        panel.title = "Save Output Image"
        panel.nameFieldStringValue = "VeraRetouch_Output.png"
        panel.allowedContentTypes = [.png]
        panel.canCreateDirectories = true

        guard panel.runModal() == .OK, let url = panel.url else { return }
        let bitmapRep = NSBitmapImageRep(cgImage: cgImage)
        guard let data = bitmapRep.representation(using: .png, properties: [:]) else {
            showSaveResult("Failed to encode image.")
            return
        }
        do {
            try data.write(to: url, options: .atomic)
            showSaveResult("Saved to \(url.lastPathComponent).")
        } catch {
            showSaveResult("Save failed: \(error.localizedDescription)")
        }
    }
    #endif
}

// MARK: - Components
struct CleanParamRow: View {
    let key: String
    @Binding var params: [String: Double]
    var range: ClosedRange<Double> = -100 ... 100

    var body: some View {
        let label = key.replacingOccurrences(of: "2012", with: "")
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(label)
                    .font(AppTypography.font(size: 12, weight: .medium))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                Spacer()
                Text("\(params[key] ?? 0.0, specifier: "%.2f")")
                    .font(AppTypography.font(size: 12, weight: .semibold))
                Button {
                    params[key] = 0.0
                } label: {
                    Image(systemName: "arrow.counterclockwise")
                        .font(AppTypography.font(size: 12, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 24, height: 24)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            Slider(
                value: Binding(
                    get: { params[key] ?? 0.0 },
                    set: { params[key] = $0 }
                ),
                in: range
            )
            .controlSize(.regular)
            .frame(height: 30)
        }
        .padding(.vertical, 6)
    }
}

struct ImageContainerView<Content: View>: View {
    let platformImage: PlatformImage?
    let cgImage: CGImage?
    let minHeight: CGFloat
    @ViewBuilder let placeholder: Content

    private var imageAspectRatio: CGFloat? {
        #if os(macOS)
        if let img = platformImage, img.size.height > 0 {
            return img.size.width / img.size.height
        }
        #else
        if let img = platformImage, img.size.height > 0 {
            return img.size.width / img.size.height
        }
        #endif
        if let cg = cgImage, cg.height > 0 {
            return CGFloat(cg.width) / CGFloat(cg.height)
        }
        return nil
    }

    var body: some View {
        let aspect = imageAspectRatio ?? 1.5
        ZStack {
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.secondary.opacity(0.2), style: StrokeStyle(lineWidth: 1, dash: [4]))
                .background(Color.clear)

            if let img = platformImage {
                #if os(macOS)
                Image(nsImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .cornerRadius(8)
                    .padding(5)
                #else
                Image(uiImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .cornerRadius(8)
                    .padding(5)
                #endif
            } else if let cg = cgImage {
                Image(decorative: cg, scale: 1.0)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .cornerRadius(8)
                    .padding(5)
            } else {
                placeholder
                    .font(AppTypography.font(size: 11))
                    .foregroundColor(.secondary)
            }
        }
        .aspectRatio(aspect, contentMode: .fit)
        .frame(minHeight: minHeight)
        .frame(maxWidth: .infinity)
    }
}

#if os(iOS)
struct LivePhotoPickerView: UIViewControllerRepresentable {
    let onPick: (String?) -> Void

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration(photoLibrary: .shared())
        config.selectionLimit = 1
        config.filter = .livePhotos
        config.preferredAssetRepresentationMode = .current
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onPick: onPick)
    }

    final class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let onPick: (String?) -> Void

        init(onPick: @escaping (String?) -> Void) {
            self.onPick = onPick
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            let localId = results.first?.assetIdentifier
            picker.dismiss(animated: true)
            onPick(localId)
        }
    }
}

struct LivePhotoPreviewView: UIViewRepresentable {
    let livePhoto: PHLivePhoto
    let replayToken: Int
    var autoPlay: Bool = true

    func makeUIView(context: Context) -> PHLivePhotoView {
        let view = PHLivePhotoView()
        view.contentMode = .scaleAspectFit
        view.isMuted = true
        view.clipsToBounds = true
        return view
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func updateUIView(_ uiView: PHLivePhotoView, context: Context) {
        let size = uiView.bounds.size
        if size.width > 1, size.height > 1, context.coordinator.lastSize != size {
            context.coordinator.lastSize = size
            context.coordinator.needsRebind = true
        }

        if uiView.livePhoto !== livePhoto || context.coordinator.needsRebind {
            context.coordinator.needsRebind = false
            context.coordinator.didAutoPlayCurrentPhoto = false
            context.coordinator.bindVersion &+= 1
            let bindVersion = context.coordinator.bindVersion
            uiView.stopPlayback()
            uiView.livePhoto = nil
            let coordinator = context.coordinator
            DispatchQueue.main.async {
                guard coordinator.bindVersion == bindVersion else { return }
                uiView.livePhoto = livePhoto
            }
        }

        if context.coordinator.lastReplayToken != replayToken {
            context.coordinator.lastReplayToken = replayToken
            uiView.startPlayback(with: .full)
            return
        }

        if autoPlay && !context.coordinator.didAutoPlayCurrentPhoto {
            context.coordinator.didAutoPlayCurrentPhoto = true
            uiView.startPlayback(with: .full)
        }
    }

    final class Coordinator {
        var lastReplayToken: Int = -1
        var didAutoPlayCurrentPhoto: Bool = false
        var lastSize: CGSize = .zero
        var needsRebind: Bool = false
        var bindVersion: Int = 0
    }
}

struct StylePromptTextView: UIViewRepresentable {
    @Binding var text: String
    @Binding var isDefault: Bool
    let defaultText: String
    let fontName: String

    func makeUIView(context: Context) -> UITextView {
        let tv = UITextView()
        tv.backgroundColor = .clear
        tv.delegate = context.coordinator
        tv.textContainerInset = UIEdgeInsets(top: 8, left: 4, bottom: 8, right: 4)
        tv.autocorrectionType = .yes
        tv.autocapitalizationType = .sentences
        if let f = UIFont(name: fontName, size: 12) {
            tv.font = f
        } else {
            tv.font = .systemFont(ofSize: 12)
        }
        tv.text = text
        tv.textColor = isDefault ? .secondaryLabel : .label
        return tv
    }

    func updateUIView(_ uiView: UITextView, context: Context) {
        if uiView.text != text {
            uiView.text = text
        }
        uiView.textColor = isDefault ? .secondaryLabel : .label
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    final class Coordinator: NSObject, UITextViewDelegate {
        var parent: StylePromptTextView

        init(_ parent: StylePromptTextView) {
            self.parent = parent
        }

        func textViewDidBeginEditing(_ textView: UITextView) {
            if parent.isDefault {
                textView.selectedRange = NSRange(location: 0, length: textView.text.count)
            }
        }

        func textViewDidChange(_ textView: UITextView) {
            let newText = textView.text ?? ""
            if parent.text != newText {
                parent.text = newText
            }
            let nowDefault = (newText == parent.defaultText)
            if parent.isDefault != nowDefault {
                parent.isDefault = nowDefault
            }
            textView.textColor = parent.isDefault ? .secondaryLabel : .label
        }
    }
}
#endif

#Preview {
    ContentView()
}
