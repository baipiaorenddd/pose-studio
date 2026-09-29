import SwiftUI

// MARK: - 叠加层画布
//
// 关键：把 Vision 的归一化坐标（原点左下）映射到画面。
// 预览层用 .resizeAspectFill，这里的映射公式与之完全一致，保证骨骼贴合人体。

struct OverlayCanvas: View {
    let poses: [Pose]
    let videoSize: CGSize
    @ObservedObject var settings: Settings

    private static let palette: [Color] = [
        Color(red: 0.00, green: 0.88, blue: 0.54),
        Color(red: 0.24, green: 0.65, blue: 1.00),
        Color(red: 1.00, green: 0.69, blue: 0.13),
        Color(red: 1.00, green: 0.34, blue: 0.47),
    ]

    var body: some View {
        Canvas { ctx, size in
            draw(ctx: ctx, size: size)
        }
        .allowsHitTesting(false)
        .ignoresSafeArea()
    }

    // MARK: 绘制

    private func draw(ctx: GraphicsContext, size: CGSize) {
        if settings.dimBackground {
            ctx.fill(Path(CGRect(origin: .zero, size: size)),
                     with: .color(Color.black.opacity(0.55)))
        }

        guard videoSize.width > 1, videoSize.height > 1, !poses.isEmpty else { return }

        // aspect-fill 映射
        let scale = max(size.width / videoSize.width, size.height / videoSize.height)
        let drawW = videoSize.width * scale
        let drawH = videoSize.height * scale
        let originX = (size.width - drawW) / 2
        let originY = (size.height - drawH) / 2

        // Vision 归一化坐标 -> 视图坐标。
        // 先按 rotationOverride 做一次兜底旋转，再做 aspect-fill 映射。
        // 正常情况下 rotationOverride = 0（自动对齐已经对了），
        // 万一设备方向判断有偏差，用户可以在界面上手动纠正。
        let rot = settings.rotationOverride
        func screen(_ p: CGPoint) -> CGPoint {
            var u = p.x
            var v = p.y
            switch rot {
            case 90:  u = p.y;       v = 1.0 - p.x
            case 180: u = 1.0 - p.x; v = 1.0 - p.y
            case 270: u = 1.0 - p.y; v = p.x
            default:  break
            }
            return CGPoint(x: originX + u * drawW,
                           y: originY + (1.0 - v) * drawH)
        }

        let lw = max(1, CGFloat(settings.lineWidth))
        let radius = max(1.5, CGFloat(settings.jointRadius))
        let kconf = Float(settings.jointConfidence)

        for (index, pose) in poses.enumerated() {
            let base = Self.palette[index % Self.palette.count]

            // ---------- 骨架连线 ----------
            if settings.showSkeleton {
                for (a, b, group) in PoseEstimator.connections {
                    guard let ja = pose.points[a], let jb = pose.points[b] else { continue }
                    if settings.hideLowConfidence && (ja.confidence < kconf || jb.confidence < kconf) { continue }
                    if !settings.showFacePoints &&
                        (PoseEstimator.faceJointIndices.contains(a) || PoseEstimator.faceJointIndices.contains(b)) { continue }

                    var path = Path()
                    path.move(to: screen(ja.position))
                    path.addLine(to: screen(jb.position))

                    let color = settings.groupColors ? groupColor(group) : base
                    ctx.stroke(path, with: .color(color),
                               style: StrokeStyle(lineWidth: lw, lineCap: .round))
                    ctx.stroke(path, with: .color(Color.white.opacity(0.85)),
                               style: StrokeStyle(lineWidth: max(1, lw - 2), lineCap: .round))
                }
            }

            // ---------- 关节圆点 ----------
            if settings.showKeypoints {
                for (i, maybe) in pose.points.enumerated() {
                    guard let j = maybe else { continue }
                    if settings.hideLowConfidence && j.confidence < kconf { continue }
                    if !settings.showFacePoints && PoseEstimator.faceJointIndices.contains(i) { continue }

                    let p = screen(j.position)
                    let outer = Path(ellipseIn: CGRect(x: p.x - radius - 1.5, y: p.y - radius - 1.5,
                                                      width: (radius + 1.5) * 2, height: (radius + 1.5) * 2))
                    ctx.fill(outer, with: .color(.white))
                    let inner = Path(ellipseIn: CGRect(x: p.x - radius, y: p.y - radius,
                                                      width: radius * 2, height: radius * 2))
                    ctx.fill(inner, with: .color(settings.groupColors ? jointColor(i, base) : base))
                }
            }

            // ---------- 关节名称 ----------
            if settings.showNames {
                for (i, maybe) in pose.points.enumerated() {
                    guard let j = maybe else { continue }
                    if settings.hideLowConfidence && j.confidence < kconf { continue }
                    if !settings.showFacePoints && PoseEstimator.faceJointIndices.contains(i) { continue }
                    let p = screen(j.position)
                    let label = Text(j.name)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.white)
                    ctx.draw(label, at: CGPoint(x: p.x + radius + 5, y: p.y - radius - 8),
                             anchor: .leading)
                }
            }

            // ---------- 边界框 + 标签 ----------
            let box = pose.boundingBox
            let p1 = screen(CGPoint(x: box.minX, y: box.maxY))   // 屏幕左上
            let p2 = screen(CGPoint(x: box.maxX, y: box.minY))   // 屏幕右下
            let pad: CGFloat = 14
            let rect = CGRect(x: min(p1.x, p2.x) - pad,
                              y: min(p1.y, p2.y) - pad,
                              width: abs(p2.x - p1.x) + pad * 2,
                              height: abs(p2.y - p1.y) + pad * 2)

            if settings.showBox {
                if settings.cornerBox {
                    drawCornerBox(ctx: ctx, rect: rect, color: base, lw: lw)
                } else {
                    ctx.stroke(Path(rect), with: .color(base), lineWidth: lw)
                }
            }

            if settings.showLabel || settings.showIDs {
                var parts: [String] = []
                if settings.showLabel { parts.append("人 \(Int(pose.confidence * 100))%") }
                if settings.showIDs { parts.append("#\(pose.id + 1)") }
                drawBadge(ctx: ctx, text: parts.joined(separator: "  "),
                          at: CGPoint(x: rect.minX, y: max(rect.minY - 26, 6)), color: base)
            }
        }
    }

    // MARK: 组件

    private func drawCornerBox(ctx: GraphicsContext, rect: CGRect, color: Color, lw: CGFloat) {
        let lx = max(rect.width * 0.22, 14)
        let ly = max(rect.height * 0.22, 14)
        var path = Path()

        path.move(to: CGPoint(x: rect.minX, y: rect.minY + ly))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.minX + lx, y: rect.minY))

        path.move(to: CGPoint(x: rect.maxX - lx, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + ly))

        path.move(to: CGPoint(x: rect.minX, y: rect.maxY - ly))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX + lx, y: rect.maxY))

        path.move(to: CGPoint(x: rect.maxX - lx, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - ly))

        ctx.stroke(path, with: .color(color),
                   style: StrokeStyle(lineWidth: lw + 1, lineCap: .round, lineJoin: .round))
        ctx.stroke(Path(rect), with: .color(color.opacity(0.35)), lineWidth: 1)
    }

    private func drawBadge(ctx: GraphicsContext, text: String, at point: CGPoint, color: Color) {
        let resolved = ctx.resolve(
            Text(text)
                .font(.system(size: 12, weight: .bold))
                .foregroundColor(.black)
        )
        let size = resolved.measure(in: CGSize(width: 400, height: 200))
        let rect = CGRect(x: point.x, y: point.y,
                          width: size.width + 14, height: size.height + 7)
        ctx.fill(Path(roundedRect: rect, cornerRadius: 5), with: .color(color))
        ctx.draw(resolved, at: CGPoint(x: rect.midX, y: rect.midY), anchor: .center)
    }

    private func groupColor(_ group: String) -> Color {
        switch group {
        case "head":  return Color(red: 0.20, green: 0.80, blue: 1.00)
        case "torso": return Color(red: 0.00, green: 0.90, blue: 0.55)
        case "armL":  return Color(red: 1.00, green: 0.80, blue: 0.10)
        case "armR":  return Color(red: 1.00, green: 0.45, blue: 0.10)
        case "legL":  return Color(red: 0.65, green: 0.45, blue: 1.00)
        default:      return Color(red: 1.00, green: 0.35, blue: 0.70)
        }
    }

    private func jointColor(_ index: Int, _ base: Color) -> Color {
        if PoseEstimator.faceJointIndices.contains(index) { return groupColor("head") }
        if [6, 8, 10].contains(index) { return groupColor("armL") }
        if [7, 9, 11].contains(index) { return groupColor("armR") }
        if [12, 14, 16].contains(index) { return groupColor("legL") }
        if [13, 15, 17].contains(index) { return groupColor("legR") }
        return base
    }
}
