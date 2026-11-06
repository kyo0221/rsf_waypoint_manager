import math
import os
from dataclasses import dataclass, field, replace

import numpy as np
import yaml
from scipy import ndimage


class Lane:

    def __init__(self, points):
        self.points = np.array(points, dtype=float)
        seg = np.diff(self.points, axis=0)
        self.lengths = np.hypot(seg[:, 0], seg[:, 1])
        keep = self.lengths > 1e-9
        self.starts = self.points[:-1][keep]
        self.lengths = self.lengths[keep]
        self.units = seg[keep] / self.lengths[:, None]
        self.offsets = np.concatenate([[0.0], np.cumsum(self.lengths)])[:-1]
        self.length = float(self.lengths.sum())

    def project_many(self, xy):
        """各点の (弧長 s, 符号付き横距離 [左が正]) を返す"""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        d = xy[:, None, :] - self.starts[None, :, :]
        t = np.clip(np.einsum('nij,ij->ni', d, self.units), 0.0, self.lengths)
        foot = self.starts[None, :, :] + self.units[None, :, :] * t[:, :, None]
        diff = xy[:, None, :] - foot
        dist = np.hypot(diff[..., 0], diff[..., 1])
        k = np.argmin(dist, axis=1)
        n = np.arange(len(xy))
        u = self.units[k]
        side = u[:, 0] * diff[n, k, 1] - u[:, 1] * diff[n, k, 0]
        return self.offsets[k] + t[n, k], np.copysign(dist[n, k], side)

    def project(self, x, y):
        s, lateral = self.project_many([[x, y]])
        return float(s[0]), float(lateral[0])

    def segment(self, s):
        return int(np.clip(np.searchsorted(self.offsets, s, side='right') - 1, 0, len(self.lengths) - 1))

    def point(self, s, lateral=0.0):
        k = self.segment(s)
        u = self.units[k]
        p = self.starts[k] + u * (s - self.offsets[k]) + np.array([-u[1], u[0]]) * lateral
        return float(p[0]), float(p[1])

    def heading(self, s):
        u = self.units[self.segment(s)]
        return math.atan2(u[1], u[0])


def route_lane(route, x, y, ahead, hint=None, before=3, after=10):
    """ロボット位置 (x, y) の直前の route 点から ahead[m] 先までを Lane にする (hint の waypoint 前後だけから探す)"""
    pts = np.array(route, dtype=float)
    if hint is not None and 0 <= hint < len(pts):
        pts = pts[max(0, hint - before):min(len(pts), hint + after)]
        if len(pts) < 2:
            pts = np.array(route, dtype=float)
    seg_lane = Lane(pts)
    s, _ = seg_lane.project(x, y)
    k = seg_lane.segment(s)
    end = k + 1
    while end < len(pts) - 1 and seg_lane.offsets[min(end, len(seg_lane.offsets) - 1)] < s + ahead:
        end += 1
    return Lane(pts[k:end + 1])


def horizontal_width(xy):
    """点群の水平の広がり [m]: 0/45/90/135 度の4方向へ射影した幅の最大 (向きによらず直径に近い)"""
    best = 0.0
    for a in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
        d = xy[:, 0] * math.cos(a) + xy[:, 1] * math.sin(a)
        best = max(best, float(d.max() - d.min()))
    return best


def is_cone_shape(span, width, max_span, max_width):
    """高さの幅 span と水平の広がり width がどちらも上限以下ならコーンの形. 上限が 0 以下なら無効 (常に False)"""
    if max_span <= 0.0 or max_width <= 0.0:
        return False
    return span <= max_span and width <= max_width


def exclude_cones(tracks):
    """コーン形状と判定された track を除いたもの (追従相手の候補用)"""
    return [tr for tr in tracks if not tr.cone]


def cluster_points(xy, cell=0.2, min_points=3, z=None):
    """xy をクラスタにして (x, y, 点数) を返す. z を渡すと (x, y, 点数, 高さの幅, 水平の広がり) を返す"""
    if len(xy) == 0:
        return []
    ij = np.floor(xy / cell).astype(int)
    lo = ij.min(axis=0)
    ij -= lo
    grid = np.zeros(ij.max(axis=0) + 1, dtype=int)
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1)
    labels, count = ndimage.label(grid > 0, structure=np.ones((3, 3)))
    point_labels = labels[ij[:, 0], ij[:, 1]]
    clusters = []
    for k in range(1, count + 1):
        member = xy[point_labels == k]
        if len(member) < min_points:
            continue
        c = (float(member[:, 0].mean()), float(member[:, 1].mean()), len(member))
        if z is not None:
            mz = z[point_labels == k]
            c += (float(mz.max() - mz.min()), horizontal_width(member))
        clusters.append(c)
    return clusters


@dataclass
class Track:
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    t: float = 0.0
    hits: int = 1
    s: float = 0.0
    history: list = field(default_factory=list)
    reliable: bool = False
    span: float = 0.0
    cone_count: int = 0
    cone: bool = False


class Tracker:
    """最近傍で対応付け, 速度は window 秒の位置履歴への直線当てはめで求める (短い履歴, 大きな残差, 速すぎる track は速度不明)"""

    def __init__(self, gate=0.8, max_age=1.0, window=1.0, min_span=0.6, max_residual=0.15, max_speed=2.5,
                 cone_max_span=0.85, cone_max_width=0.45, cone_frames=2):
        self.cone_max_span, self.cone_max_width, self.cone_frames = cone_max_span, cone_max_width, cone_frames
        self.gate, self.max_age = gate, max_age
        self.window, self.min_span, self.max_residual, self.max_speed = window, min_span, max_residual, max_speed
        self.tracks = []
        self.next_id = 0

    def clear(self):
        """追跡をすべて捨てる. ID は続きから振る (前の相手と同じ ID にしない)"""
        self.tracks = []

    def fit(self, tr):
        h = [p for p in tr.history if p[0] >= tr.t - self.window]
        tr.history = h
        tr.vx = tr.vy = 0.0
        tr.reliable = False
        tr.span = h[-1][0] - h[0][0]
        if tr.span < self.min_span:
            return
        a = np.array(h)
        tt = a[:, 0] - a[:, 0].mean()
        denom = float(tt @ tt)
        vx, vy = float(tt @ a[:, 1]) / denom, float(tt @ a[:, 2]) / denom
        rx = a[:, 1] - a[:, 1].mean() - vx * tt
        ry = a[:, 2] - a[:, 2].mean() - vy * tt
        if math.sqrt(float(rx @ rx + ry @ ry) / len(a)) > self.max_residual or math.hypot(vx, vy) > self.max_speed:
            return
        tr.vx, tr.vy, tr.reliable = vx, vy, True

    def mark_cone(self, tr, c):
        """対応したクラスタ c (x, y, n[, 高さの幅, 広がり]) がコーン形状なら連続回数を増やし, 違えば 0 に戻す (人を誤って除外しない)"""
        shaped = len(c) > 4 and is_cone_shape(c[3], c[4], self.cone_max_span, self.cone_max_width)
        tr.cone_count = tr.cone_count + 1 if shaped else 0
        tr.cone = tr.cone_count >= self.cone_frames

    def update(self, t, centroids):
        free = list(self.tracks)
        for c in centroids:
            x, y = c[0], c[1]
            best = None
            for tr in free:
                dt = t - tr.t
                d = math.hypot(x - (tr.x + tr.vx * dt), y - (tr.y + tr.vy * dt))
                if d < self.gate and (best is None or d < best[0]):
                    best = (d, tr)
            if best is None:
                tr = Track(self.next_id, x, y, t=t, history=[(t, x, y)])
                self.mark_cone(tr, c)
                self.tracks.append(tr)
                self.next_id += 1
                continue
            tr = best[1]
            free.remove(tr)
            tr.x, tr.y, tr.t, tr.hits = x, y, t, tr.hits + 1
            tr.history.append((t, x, y))
            self.mark_cone(tr, c)
            self.fit(tr)
        self.tracks = [tr for tr in self.tracks if t - tr.t <= self.max_age]
        return list(self.tracks)


def tracks_in_frame(tracks, tx, ty, yaw):
    """track を剛体変換 (平行移動 tx, ty, 回転 yaw) した別座標系での複製を返す"""
    c, s = math.cos(yaw), math.sin(yaw)
    return [replace(tr, x=tx + c * tr.x - s * tr.y, y=ty + s * tr.x + c * tr.y,
                    vx=c * tr.vx - s * tr.vy, vy=s * tr.vx + c * tr.vy) for tr in tracks]


def find_oncoming(lane, robot_s, tracks, half_width, ahead, min_speed, min_hits, min_ahead=0.0, now=None,
                  max_stale=0.3, min_travel=0.0):
    """lane 上で前方 min_ahead〜ahead[m], 横 half_width 以内をこちらへ向かう最も近い track (min_travel: 履歴の間に近づいた距離の下限)"""
    best = None
    for tr in tracks:
        if tr.hits < min_hits or not tr.reliable or (now is not None and now - tr.t > max_stale):
            continue
        s, lateral = lane.project(tr.x, tr.y)
        if not max(min_ahead, 0.0) < s - robot_s <= ahead or abs(lateral) > half_width:
            continue
        h = lane.heading(s)
        closing = -(tr.vx * math.cos(h) + tr.vy * math.sin(h))
        if closing < min_speed or closing * tr.span < min_travel:
            continue
        if best is None or s < best.s:
            tr.s = s
            best = tr
    return best


def find_leader(lane, robot_s, tracks, half_width, ahead, min_speed, min_hits, now=None, max_stale=0.3,
                min_travel=0.0):
    """lane 上で前方 ahead[m], 横 half_width 以内を同じ向きに min_speed 以上で進む最も近い track (追従相手)"""
    best = None
    for tr in tracks:
        if tr.hits < min_hits or not tr.reliable or (now is not None and now - tr.t > max_stale):
            continue
        s, lateral = lane.project(tr.x, tr.y)
        if not 0.0 < s - robot_s <= ahead or abs(lateral) > half_width:
            continue
        h = lane.heading(s)
        forward = tr.vx * math.cos(h) + tr.vy * math.sin(h)
        if forward < min_speed or forward * tr.span < min_travel:
            continue
        if best is None or s < best.s:
            tr.s = s
            best = tr
    return best


def follow_speed(gap, standoff, time_gap, max_speed, floor=0.001):
    """前方 gap[m] の相手に standoff を保って追従する速度上限. 速度上限 0 は「制限なし」扱いなので floor 以上を返す"""
    return min(max_speed, max(floor, (gap - standoff) / time_gap))


class YieldStateMachine:
    """idle -> pausing -> moving -> waiting -> idle. step() は実行すべき動作 ('pause', 'goto', 'resume') を返す"""

    def __init__(self, clear_time, max_wait, cooldown, pause_timeout=5.0, confirm_time=0.0):
        self.clear_time, self.max_wait = clear_time, max_wait
        self.cooldown, self.pause_timeout, self.confirm_time = cooldown, pause_timeout, confirm_time
        self.oncoming_since = None
        self.state = 'idle'
        self.since = 0.0
        self.last_oncoming = -math.inf
        self.blocked_until = -math.inf

    def enter(self, state, now):
        self.state, self.since = state, now

    def step(self, now, enabled, running, paused, oncoming, goal_done, waiting_for=False):
        """waiting_for: 待避の原因になった物体がまだ前方 (真横を過ぎていない) にいる"""
        if oncoming or (waiting_for and self.state != 'idle'):
            self.last_oncoming = now
        self.oncoming_since = (self.oncoming_since if self.oncoming_since is not None else now) if oncoming else None
        if self.state == 'idle':
            confirmed = oncoming and now - self.oncoming_since >= self.confirm_time
            if enabled and running and confirmed and now >= self.blocked_until:
                self.enter('pausing', now)
                return 'pause'
            return None
        if self.state == 'pausing':
            if paused:
                self.enter('moving', now)
                return 'goto'
            if now - self.since > self.pause_timeout:
                self.enter('idle', now)
                self.blocked_until = now + self.cooldown
            return None
        if self.state == 'moving' and goal_done:
            self.enter('waiting', now)
        waited_too_long = self.state == 'waiting' and now - self.since > self.max_wait
        if now - self.last_oncoming >= self.clear_time or waited_too_long:
            self.enter('idle', now)
            self.blocked_until = now + self.cooldown
            return 'resume'
        return None


def next_waypoint_index(route, stops, robot_xy, current, margin=1.0, window=6):
    """待機後に向かう waypoint: current 付近だけを見て, ロボットより margin[m] 以上先の最初の waypoint (停止点は飛ばさない)"""
    lo, hi = max(0, current - 1), min(len(route) - 1, current + window)
    if hi <= lo:
        return current
    lane = Lane(route[lo:hi + 1])
    robot_s, _ = lane.project(*robot_xy)
    offsets = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(np.array(route[lo:hi + 1], dtype=float), axis=0).T))])
    target = hi
    for k in range(current, hi + 1):
        if offsets[k - lo] - robot_s > margin:
            target = k
            break
    for k in range(current, target):
        if stops[k]:
            return k
    return target


class LeaderMonitor:
    """追従相手が release_after 秒以上動かなければ手放し, 手放した相手は再び追わない"""

    def __init__(self, release_after, moving_speed):
        self.release_after, self.moving_speed = release_after, moving_speed
        self.moved_at = {}
        self.released = set()

    def update(self, now, leader_id, speed):
        if leader_id in self.released:
            return False
        if speed >= self.moving_speed or leader_id not in self.moved_at:
            self.moved_at[leader_id] = now
        if now - self.moved_at[leader_id] > self.release_after:
            self.released.add(leader_id)
            return False
        return True


def should_resend_refuge(status, elapsed, retries, canceled_by_us, max_retries=3, early=2.0):
    """待機点ゴールが送ってすぐ (early 秒以内) に取り消し/中断されたら送り直す"""
    if canceled_by_us or retries >= max_retries or elapsed > early:
        return False
    return status in (5, 6)  # action_msgs/GoalStatus: CANCELED, ABORTED


def obstacle_layer_action(following, layer_off, follow_enabled):
    """追従中は障害物層を切り, やめたら戻す ('disable'/'enable'). follow_enabled が false なら層はプロファイル管理なので 'forget'"""
    if following and not layer_off:
        return 'disable'
    if not following and layer_off:
        return 'enable' if follow_enabled else 'forget'
    return None


def smooth_limit(prev, target, dt, up, down):
    """速度上限 prev を target へ, 上げは up, 下げは down [m/s^2] 以内の変化で近づける (prev=None は target)"""
    if prev is None:
        return target
    return min(prev + up * dt, target) if target > prev else max(prev - down * dt, target)


def predict_track(track, now, max_dt):
    """最後に見えた位置を速度で最大 max_dt[s] 進めた複製 (t は最後に見えた時刻のまま)"""
    dt = min(max(0.0, now - track.t), max_dt)
    return replace(track, x=track.x + track.vx * dt, y=track.y + track.vy * dt)


def find_refuge(lane, robot_xy, opponent_xy, free, path_clear, ahead=3.0, lateral=(1.0, 2.0), step=0.5,
                sides=(1.0, -1.0), min_sep=0.0, behind=0.0):
    """前方斜め前 (出会う地点の手前), 無ければ真横, 後方 (behind[m] まで) の順に free と path_clear を満たす待機点 (x, y, yaw) を探す"""
    robot_s, robot_l = lane.project(*robot_xy)
    opponent_s, opponent_l = lane.project(*opponent_xy)
    limit = min(ahead, (opponent_s - robot_s) / 2.0 - 0.5)
    for forward in (np.arange(step, limit + 1e-6, step), np.array([0.0]), -np.arange(step, behind + 1e-6, step)):
        best = None
        for ds in forward:
            s = robot_s + ds
            for side in sides:
                for dl in np.arange(lateral[0], lateral[1] + 1e-6, step):
                    l = robot_l + side * dl
                    p = lane.point(s, l)
                    if abs(l - opponent_l) < min_sep or not free(p) or not path_clear(robot_xy, p):
                        continue
                    # 前方は向きの変化が小さい点, 真横と後方は移動が短い点
                    tie = math.atan2(dl, ds) if ds > 0 else math.hypot(ds, dl)
                    key = (-round(abs(l - opponent_l), 2), round(tie, 3))
                    if best is None or key < best[0]:
                        best = (key, (p[0], p[1], lane.heading(s)))
        if best is not None:
            return best[1]
    return None


def align_angle(current, target, tol):
    """current から target へ向きをそろえる回転量 [rad] (短い方). tol 以内なら None"""
    d = math.atan2(math.sin(target - current), math.cos(target - current))
    return None if abs(d) <= tol else d


def points_near(points, xy, radius):
    """xy から radius 以内に点があれば True"""
    if len(points) == 0:
        return False
    return bool(np.any(np.hypot(points[:, 0] - xy[0], points[:, 1] - xy[1]) <= radius))


def choose_fixed_refuge(lane, robot_xy, opponent_xy, points, ahead, check=None, max_lateral=3.0):
    """固定の退避点のうち, 前方 ahead[m] 以内かつ出会う地点の手前で check を満たす最も近い点 (x, y, yaw). 横に max_lateral 超は除く"""
    if not points:
        return None
    robot_s, robot_l = lane.project(*robot_xy)
    opponent_s, _ = lane.project(*opponent_xy)
    limit = min(ahead, (opponent_s - robot_s) / 2.0 - 0.5)
    pts = np.asarray(points, dtype=float)
    s, lateral = lane.project_many(pts)
    order = np.argsort(s)
    for i in order:
        if not (robot_s < s[i] <= robot_s + limit) or abs(lateral[i] - robot_l) > max_lateral:
            continue
        p = (float(pts[i, 0]), float(pts[i, 1]))
        if check is None or check(p):
            return (p[0], p[1], lane.heading(float(s[i])))
    return None


# waypoint の上の物体とみなす半径 [m]. 0.6 だと経路の横 0.7m のコーン列 (表面は 0.52m) を拾って止まる
SKIP_RADIUS = 0.45


def should_skip_waypoint(wp_xy, is_stop, robot_xy, points, radius, within):
    """waypoint の radius[m] 以内に物体の点があり, ロボットが within[m] 以内まで来ていれば True (停止点は除く)"""
    if is_stop or math.hypot(wp_xy[0] - robot_xy[0], wp_xy[1] - robot_xy[1]) > within:
        return False
    return points_near(points, wp_xy, radius)


def find_retreat(lane, robot_xy, free, dists=(1.5, 2.0, 2.5, 3.0), laterals=(0.0,)):
    """最後の手段: レーンを dists[m] 後ろへ下がる点のうち free を満たす最も近い点 (x, y, yaw. 向きはレーンのまま)"""
    s, _ = lane.project(*robot_xy)
    for d in dists:
        for lat in laterals:
            p = lane.point(s - d, lat)
            if free(p):
                return (p[0], p[1], lane.heading(s - d))
    return None


def stop_ahead_within(route, stops, k, within):
    """waypoint k から経路に沿って within[m] 以内 (k を含む) にある最初の停止点の番号. 無ければ None"""
    dist = 0.0
    for j in range(k, len(route)):
        if j > k:
            dist += math.hypot(route[j][0] - route[j - 1][0], route[j][1] - route[j - 1][1])
        if dist > within:
            return None
        if stops[j]:
            return j
    return None


def near_stop_ahead(route, stops, k, within):
    """waypoint k から経路に沿って within[m] 以内 (k を含む) に停止点があれば True"""
    return stop_ahead_within(route, stops, k, within) is not None


def stalled(history, now, window, min_move):
    """history [(t, (x, y))] で直近 window[s] 全体を通して移動が min_move[m] 未満なら True"""
    recent = [h for h in history if h[0] >= now - window - 1.0]
    if not recent or now - recent[0][0] < window:
        return False
    x0, y0 = recent[-1][1]
    return max(math.hypot(x - x0, y - y0) for _, (x, y) in recent) < min_move


def static_hits(xy, info, mask):
    """map 座標の点 xy のうち, 静的マスク mask (info: OccupancyGrid の info) の占有セルに入るものを True にした配列"""
    mi = np.floor((xy[:, 0] - info.origin.position.x) / info.resolution).astype(int)
    mj = np.floor((xy[:, 1] - info.origin.position.y) / info.resolution).astype(int)
    ok = (mi >= 0) & (mi < info.width) & (mj >= 0) & (mj < info.height)
    hit = np.zeros(len(xy), dtype=bool)
    hit[ok] = mask[mj[ok], mi[ok]]
    return hit


def split_obstacles(xy, z, lane_keep, hit):
    """追従・対向の判定用 (経路の帯の中で地図の固定物でない点) の xy, z と, 詰まりの判定用 (固定物も含む全点) の xy"""
    keep = lane_keep & ~hit
    return xy[keep], z[keep], xy


def blocked_ahead(points, pose, ahead, half_width):
    """map 座標の点のうち, ロボット (x, y, yaw) の正面 ahead[m] 以内・横 half_width[m] 以内にあるものがあれば True"""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    for px, py in points:
        fx, fy = c * (px - x) + s * (py - y), -s * (px - x) + c * (py - y)
        if 0.0 < fx <= ahead and abs(fy) <= half_width:
            return True
    return False


def backup_distance(last, now, xy, step=0.4, limit=1.2, near=1.0, within=60.0):
    """後退する距離と記録 (時刻, 位置, 回数). 前回と同じ場所 (near[m] 以内, within[s] 以内) でまた詰まったら step ずつ伸ばす"""
    count = 1
    if last is not None and now - last[0] <= within and math.hypot(xy[0] - last[1][0], xy[1] - last[1][1]) <= near:
        count = last[2] + 1
    return min(step * count, limit), (now, xy, count)


# ---- 退避禁止の範囲 (横断歩道と道路端の間の車道) ----

def point_in_polygon(p, poly):
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def zones_at(p, zones):
    """p を含む範囲の添字の集合"""
    return {i for i, z in enumerate(zones) if point_in_polygon(p, z['points'])}


def in_zones(p, zones):
    """p がどれかの範囲 (4 隅の多角形) の中にあれば True"""
    return any(point_in_polygon(p, z['points']) for z in zones)


def segment_hits_zones(a, b, zones, step=0.1, allow_exit=False):
    """a から b への直線 (両端を含む, step[m] 刻み) が範囲を通れば True.
    allow_exit: a がいる範囲の中を進むのと, そこから外へ出る向きは許す (別の範囲や外から戻った範囲は不可)"""
    if not zones:
        return False
    n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
    home = zones_at(a, zones) if allow_exit else set()
    inside_home = bool(home)
    for k in range(n + 1):
        p = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
        hit = zones_at(p, zones)
        if not hit:
            inside_home = False
            continue
        if inside_home and hit <= home:
            continue
        return True
    return False


def load_zones(path):
    """範囲の yaml を [{'name', 'type', 'points': [[x, y] x4]}] で読む. 無ければ空"""
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        raw = (yaml.safe_load(f) or {}).get('zones') or []
    return [{'name': str(z.get('name', '')), 'type': str(z.get('type', 'roadway')),
             'points': [[float(x), float(y)] for x, y in z['points']]} for z in raw]


def save_zones(path, zones):
    """realpath の先へ in-place で書く (キーの順は name/type/points)"""
    out = [{'name': z['name'], 'type': z['type'], 'points': [[float(x), float(y)] for x, y in z['points']]}
           for z in zones]
    with open(os.path.realpath(path), 'w') as f:
        yaml.safe_dump({'zones': out}, f, sort_keys=False, default_flow_style=None, allow_unicode=True)


def zones_file_for(waypoints_file, param=''):
    """範囲 yaml のパス: param が空なら <name>_wp.yaml に対する <name>_no_refuge_zones.yaml"""
    if param:
        return param
    if waypoints_file.endswith('_wp.yaml'):
        return waypoints_file[:-len('_wp.yaml')] + '_no_refuge_zones.yaml'
    return ''


def filter_points_outside_zones(points, zones):
    """(範囲の外の点, 範囲の中の点)"""
    kept = [p for p in points if not in_zones(p, zones)]
    dropped = [p for p in points if in_zones(p, zones)]
    return kept, dropped


def oncoming_for_machine(oncoming, state, robot_xy, zones):
    """状態機械へ渡す oncoming: 待避を始める前 (idle) にロボットが範囲内なら False (減速して渡り切るのを優先)"""
    if oncoming and state == 'idle' and robot_xy is not None and in_zones(robot_xy, zones):
        return False
    return bool(oncoming)


def passing_oncoming(lane, robot_s, robot_l, tracks, ahead, lateral, min_speed):
    """前方 ahead[m] 以内・横 lateral[m] 以内を min_speed[m/s] 以上でこちらへ向かってくる track があれば True (すれ違い中)"""
    for tr in tracks:
        if not tr.reliable:
            continue
        s, lat = lane.project(tr.x, tr.y)
        if not -1.0 <= s - robot_s <= ahead or abs(lat - robot_l) > lateral:
            continue
        h = lane.heading(s)
        if -(tr.vx * math.cos(h) + tr.vy * math.sin(h)) >= min_speed:
            return True
    return False
