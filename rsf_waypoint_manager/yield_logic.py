import math
from dataclasses import dataclass, field, replace

import numpy as np
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
    """ロボット位置 (x, y) の直前の route 点から ahead[m] 先までを Lane にする.
    hint (向かっている waypoint 番号) があればその前後だけから探す (往路と復路が重なる区間の取り違え防止)"""
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


def cluster_points(xy, cell=0.2, min_points=3):
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
        if len(member) >= min_points:
            clusters.append((float(member[:, 0].mean()), float(member[:, 1].mean()), len(member)))
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


class Tracker:
    """最近傍で対応付け, 速度は window 秒の位置履歴への直線当てはめで求める.
    履歴が min_span 秒未満, 残差が max_residual を超える (重心が飛ぶ), または max_speed を超える track は速度不明とする"""

    def __init__(self, gate=0.8, max_age=1.0, window=1.0, min_span=0.6, max_residual=0.15, max_speed=2.5):
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

    def update(self, t, centroids):
        free = list(self.tracks)
        for x, y in ((c[0], c[1]) for c in centroids):
            best = None
            for tr in free:
                dt = t - tr.t
                d = math.hypot(x - (tr.x + tr.vx * dt), y - (tr.y + tr.vy * dt))
                if d < self.gate and (best is None or d < best[0]):
                    best = (d, tr)
            if best is None:
                self.tracks.append(Track(self.next_id, x, y, t=t, history=[(t, x, y)]))
                self.next_id += 1
                continue
            tr = best[1]
            free.remove(tr)
            tr.x, tr.y, tr.t, tr.hits = x, y, t, tr.hits + 1
            tr.history.append((t, x, y))
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
    """lane 上でロボットの前方 min_ahead〜ahead[m], 横 half_width 以内をこちらへ向かって進む最も近い track.
    now を渡すと max_stale 秒以上観測されていない track は除く.
    min_travel: 履歴の期間にこちらへ近づいた距離の下限 (静止物の見かけの重心移動は物体の大きさ程度で頭打ちになる)"""
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
    """lane 上でロボットの前方 ahead[m], 横 half_width 以内を同じ向きに min_speed 以上で進む最も近い track (追従相手).
    min_travel: 履歴の期間に進んだ距離の下限 (静止物の見かけの重心移動を除く)"""
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
    """待機後に向かう waypoint. current 付近の区間だけを見て (往復で重なる区間の取り違え防止),
    ロボットより margin[m] 以上先にある最初の waypoint を返す. 途中の停止点は飛ばさない"""
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
    """追従相手が release_after 秒以上動かなければ手放す (静止物の取り違えで止まり続けないため). 手放した相手は再び追わない"""

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
    """待機点ゴールが送ってすぐ (early 秒以内) に取り消し/中断されたら送り直す.
    一時停止で取り消した waypoint 走行のキャンセルが, 後から送った待機点ゴールに当たることがあるため"""
    if canceled_by_us or retries >= max_retries or elapsed > early:
        return False
    return status in (5, 6)  # action_msgs/GoalStatus: CANCELED, ABORTED


def obstacle_layer_action(following, layer_off, follow_enabled):
    """追従中は global の障害物層を切り, 追従をやめたら戻す. layer_off: こちらが切った状態か.
    follow_enabled が false (queue_guard 系プロファイル) なら層はプロファイルが管理するので戻さず忘れる ('forget')"""
    if following and not layer_off:
        return 'disable'
    if not following and layer_off:
        return 'enable' if follow_enabled else 'forget'
    return None


def smooth_limit(prev, target, dt, up, down):
    """速度上限 prev を target へ, 上げは up[m/s^2], 下げは down[m/s^2] 以内の変化で近づける (prev=None はそのまま target)"""
    if prev is None:
        return target
    return min(prev + up * dt, target) if target > prev else max(prev - down * dt, target)


def predict_track(track, now, max_dt):
    """最後に見えた位置を速度で最大 max_dt[s] 進めた複製 (t は最後に見えた時刻のまま)"""
    dt = min(max(0.0, now - track.t), max_dt)
    return replace(track, x=track.x + track.vx * dt, y=track.y + track.vy * dt)


def find_refuge(lane, robot_xy, opponent_xy, free, path_clear, ahead=3.0, lateral=(1.0, 2.0), step=0.5, prefer=None,
                sides=(1.0, -1.0), min_sep=0.0, behind=0.0):
    """待避を決めた時点でロボットの前方斜め前 (前方 step〜ahead[m], 左右に lateral[m]) を step 刻みで探す.
    前方は相手との間隔の半分から 0.5m 手前まで (出会う地点より手前). free(点) と path_clear(ロボット, 点) を満たす点のうち,
    prefer(点) が True の点 (歩行者側) を優先し, 次に相手の横位置から最も離れ, 次に向きの変化が小さい (より前方の) 点を返す.
    前方に候補が無ければ真横, それも無ければ後方 (behind[m] まで) を探す. sides はレーンの左 (+1) / 右 (-1) のうち探す側.
    相手の横位置から min_sep 未満の点は相手の進路から外れきれないので使わない.
    戻り値は (x, y, yaw) か None"""
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
                    preferred = prefer is None or bool(prefer(p))
                    # 前方は向きの変化が小さい点, 真横と後方は移動が短い点
                    tie = math.atan2(dl, ds) if ds > 0 else math.hypot(ds, dl)
                    key = (not preferred, -round(abs(l - opponent_l), 2), round(tie, 3))
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
    """固定の退避点 points [(x, y)] のうち, 前方 ahead[m] 以内かつ出会う地点 (相手との間隔の半分 - 0.5m) より手前で,
    check(点) を満たす最も近い点を (x, y, yaw) で返す. レーンから横に max_lateral[m] より離れた点は使わない"""
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


def should_skip_waypoint(wp_xy, is_stop, robot_xy, points, radius, within):
    """向かっている waypoint の radius[m] 以内に物体の点があり, ロボットが within[m] 以内まで来ていれば True.
    停止点は飛ばさない (同じ場所で止まる必要がある)"""
    if is_stop or math.hypot(wp_xy[0] - robot_xy[0], wp_xy[1] - robot_xy[1]) > within:
        return False
    return points_near(points, wp_xy, radius)


def find_retreat(lane, robot_xy, free, dists=(1.5, 2.0, 2.5, 3.0), laterals=(0.0,)):
    """待避点が無いときの最後の手段: レーンの中央を dists[m] 後ろへ下がる点のうち free を満たす最も近い点 (x, y, yaw).
    yaw はレーンの向き (向きを変えずに後退で下がる). ポールの間などで対向と向き合って動けなくなるのを防ぐ"""
    s, _ = lane.project(*robot_xy)
    for d in dists:
        for lat in laterals:
            p = lane.point(s - d, lat)
            if free(p):
                return (p[0], p[1], lane.heading(s - d))
    return None
