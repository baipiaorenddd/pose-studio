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

    /// 与 AVCaptureVideoDataOutput 保持一致：竖屏 90°，镜像交给系统自动处理
    private func applyOrientation(_ view: PreviewUIView) {
        guard let conn = view.previewLayer.connection else { return }
        if conn.isVideoRotationAngleSupported(90) {
            conn.videoRotationAngle = 90
        }
    }
}
