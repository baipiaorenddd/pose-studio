import SwiftUI
import AVFoundation

// MARK: - 摄像头预览（底层 AVCaptureVideoPreviewLayer）

final class PreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.backgroundColor = .black
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        applyOrientation(view)
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        if uiView.previewLayer.session !== session {
            uiView.previewLayer.session = session
        }
        applyOrientation(uiView)
    }

    /// 方向 + 镜像，必须和 PoseEstimator 里的数据输出设置严格配套：
    ///   * 数据输出：永不镜像（automaticallyAdjustsVideoMirroring = false, isVideoMirrored = false）
    ///   * 预览层  ：前置时镜像
    ///   * 叠加层  ：前置时把 x 翻过来（overlayMirrored）
    /// 三处都显式指定，不依赖系统的"自动镜像"，就不会出现骨架左右翻的问题。
    private func applyOrientation(_ view: PreviewUIView) {
        guard let conn = view.previewLayer.connection else { return }
        if conn.isVideoRotationAngleSupported(90) {
            conn.videoRotationAngle = 90
        }

        let front = view.previewLayer.session?.inputs
            .compactMap { ($0 as? AVCaptureDeviceInput)?.device.position }
            .first == .front

        if conn.isVideoMirroringSupported {
            conn.automaticallyAdjustsVideoMirroring = false
            conn.isVideoMirrored = front
        }
    }
}
