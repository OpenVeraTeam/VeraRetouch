//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import AVFoundation
import SwiftUI

public struct CameraControlsView: View {

    @Binding public var backCamera: Bool
    @Binding public var device: AVCaptureDevice
    @Binding public var devices: [AVCaptureDevice]

    public init(
        backCamera: Binding<Bool>,
        device: Binding<AVCaptureDevice>,
        devices: Binding<[AVCaptureDevice]>
    ) {
        self._backCamera = backCamera
        self._device = device
        self._devices = devices
    }

    public var body: some View {
        Button {
            backCamera.toggle()
        } label: {
            RoundedRectangle(cornerRadius: 8.0)
                .fill(.regularMaterial)
                .frame(width: 32.0, height: 32.0)
                .overlay(alignment: .center) {
                    // Switch cameras image
                    Image(systemName: "arrow.triangle.2.circlepath.camera.fill")
                        .foregroundStyle(.primary)
                        .padding(6.0)
                }
        }
    }
}
