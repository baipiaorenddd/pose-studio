import SwiftUI
import Foundation

// MARK: - 叠加层画布
//
// 把 Vision 的归一化坐标（原点左下）映射到画面。
// 预览层用 .resizeAspectFill，这里的映射公式与之完全一致，保证骨骼贴合人体。

struct OverlayCanvas: View {
    let poses: [Pose]
    let videoSize: CGSize
    /// 前置摄像头时预览是镜像的，骨架也要跟着镜像，否则左右会反
    let mirrored: Bool
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
            if mirrored { u = 1.0 - u }
            return CGPoint(x: originX + u * drawW,
                           y: originY + (1.0 - v) * drawH)
        }

        let lw = max(1, CGFloat(settings.lineWidth))
        let kconf = Float(settings.jointConfidence)

        for (index, pose) in poses.enumerated() {
            // 分区配色关闭时 = 纯白骨架
            let base: Color = settings.groupColors
                ? Self.palette[index % Self.palette.count]
                : Color.white
            let hideFace = settings.stickHead || !settings.showFacePoints

            // ---------- 骨架连线 ----------
            if settings.showSkeleton {
                for (a, b, group) in PoseEstimator.connections {
                    guard let ja = pose.points[a], let jb = pose.points[b] else { continue }
                    if settings.hideLowConfidence && (ja.confidence < kconf || jb.confidence < kconf) { continue }
                    if hideFace &&
                        (PoseEstimator.faceJointIndices.contains(a) || PoseEstimator.faceJointIndices.contains(b)) { continue }

                    var path = Path()
                    path.move(to: screen(ja.position))
                    path.addLine(to: screen(jb.position))

                    let color = settings.groupColors ? groupColor(group) : base
                    ctx.stroke(path, with: .color(color),
                               style: StrokeStyle(lineWidth: lw, lineCap: .round))
                }
            }

            // ---------- 火柴人头部（圆圈，纯线条、不填充）----------
            if settings.stickHead, let head = headCircle(pose, screen: screen, minConf: kconf) {
                let headColor = settings.groupColors ? groupColor("head") : base
                let hr = min(max(head.radius, 6), min(size.width, size.height) * 0.25)

                // 脖子 -> 圆圈下缘的连线，补上隐藏面部关键点后留下的缺口
                if let neckJoint = pose.points[5], neckJoint.confidence >= kconf {
                    let neckPt = screen(neckJoint.position)
                    let dx = neckPt.x - head.center.x
                    let dy = neckPt.y - head.center.y
                    let dist = max(hypot(dx, dy), 0.001)
                    if dist > hr {
                        let edge = CGPoint(x: head.center.x + dx / dist * hr,
                                           y: head.center.y + dy / dist * hr)
                        var neckPath = Path()
                        neckPath.move(to: neckPt)
                        neckPath.addLine(to: edge)
                        ctx.stroke(neckPath, with: .color(headColor),
                                   style: StrokeStyle(lineWidth: lw, lineCap: .round))
                    }
                }

                let circlePath = Path(ellipseIn: CGRect(x: head.center.x - hr,
                                                        y: head.center.y - hr,
                                                        width: hr * 2,
                                                        height: hr * 2))
                if settings.headFill {
                    ctx.fill(circlePath, with: .color(headColor.opacity(0.20)))
                }
                ctx.stroke(circlePath, with: .color(headColor),
                           style: StrokeStyle(lineWidth: lw, lineCap: .round))
            }

            // ---------- 关节名称 ----------
            if settings.showNames {
                for (i, maybe) in pose.points.enumerated() {
                    guard let j = maybe else { continue }
                    if settings.hideLowConfidence && j.confidence < kconf { continue }
                    if hideFace && PoseEstimator.faceJointIndices.contains(i) { continue }
                    let p = screen(j.position)
                    let label = Text(j.name)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.white)
                    ctx.draw(label, at: CGPoint(x: p.x + 5, y: p.y - 8), anchor: .leading)
                }
            }

            // ---------- 边界框 + 标签 ----------
            let box = pose.boundingBox
            let p1 = screen(CGPoint(x: box.minX, y: box.maxY))
            let p2 = screen(CGPoint(x: box.maxX, y: box.minY))
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
                          at: CGPoint(x: rect.minX, y: max(rect.minY - 26, 6)),
                          color: base, light: !settings.groupColors)
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

    /// 白色骨架上的标签要用深色底 + 白字，否则白底白字看不见
    private func drawBadge(ctx: GraphicsContext, text: String, at point: CGPoint,
                           color: Color, light: Bool) {
        let bg: Color = light ? Color.black.opacity(0.55) : color
        let fg: Color = light ? Color.white : Color.black
        let resolved = ctx.resolve(
            Text(text)
                .font(.system(size: 12, weight: .bold))
                .foregroundColor(fg)
        )
        let size = resolved.measure(in: CGSize(width: 400, height: 200))
        let rect = CGRect(x: point.x, y: point.y,
                          width: size.width + 14, height: size.height + 7)
        ctx.fill(Path(roundedRect: rect, cornerRadius: 5), with: .color(bg))
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

    /// 由面部关键点推算一个能罩住头部的圆（火柴人头部）。
    /// 优先用双耳距离（约等于头宽），其次双眼，最后用鼻-颈距离。
    private func headCircle(_ pose: Pose,
                            screen: (CGPoint) -> CGPoint,
                            minConf: Float) -> (center: CGPoint, radius: CGFloat)? {
        func pt(_ i: Int) -> CGPoint? {
            guard let j = pose.points[i], j.confidence >= minConf else { return nil }
            return screen(j.position)
        }

        let nose = pt(0)
        let lEye = pt(1), rEye = pt(2)
        let lEar = pt(3), rEar = pt(4)
        let neck = pt(5)

        var center: CGPoint?
        var radius: CGFloat = 0

        if let a = lEar, let b = rEar {
            center = CGPoint(x: (a.x + b.x) / 2, y: (a.y + b.y) / 2)
            radius = hypot(b.x - a.x, b.y - a.y) * 0.60
        } else if let a = lEye, let b = rEye {
            center = CGPoint(x: (a.x + b.x) / 2, y: (a.y + b.y) / 2)
            radius = hypot(b.x - a.x, b.y - a.y) * 1.10
        } else if let n = nose, let k = neck {
            let d = hypot(n.x - k.x, n.y - k.y)
            center = CGPoint(x: n.x, y: n.y - d * 0.18)
            radius = d * 0.62
        } else if let n = nose {
            center = n
            radius = 28
        }

        // 3D 姿态模型没有任何面部关节，只能用双肩推算头部位置
        if center == nil, let ls = pt(6), let rs = pt(7) {
            let mid = CGPoint(x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2)
            let shoulderW = hypot(rs.x - ls.x, rs.y - ls.y)
            // 屏幕坐标 y 向下，头部在肩膀上方的 y 更小
            center = CGPoint(x: mid.x, y: mid.y - shoulderW * 0.52)
            radius = shoulderW * 0.42
        }

        guard let c = center else { return nil }
        return (c, max(radius, 8))
    }
}
