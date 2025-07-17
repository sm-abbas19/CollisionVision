import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from CollisionVision.helpers.kalman_filter import KalmanFilter
from CollisionVision.helpers import matching
from CollisionVision.helpers.basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, class_id=-1):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.class_id = class_id

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks],dtype=np.float32)
            multi_covariance = np.asarray([st.covariance for st in stracks],dtype=np.float32)
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=True):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.class_id = new_track.class_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.class_id = new_track.class_id

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy().astype(np.float32)  
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret.astype(np.float32) 

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = np.asarray(self.tlwh, dtype=np.float32).copy()
        ret[2:] += ret[:2]
        return ret.astype(np.float32)

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret.astype(np.float32)

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr, dtype=np.float32).copy()
        ret[2:] -= ret[:2]
        return ret.astype(np.float32)

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh,dtype=np.float32).copy()
        ret[2:] += ret[:2]
        return ret.astype(np.float32)

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=5):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh
        #self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        if output_results.shape[0] == 0:
            return []

        # If this is the first frame or there are no existing tracks,
        # initialize all detections as new tracks.
        if self.frame_id == 1 or (len(self.tracked_stracks) == 0 and len(self.lost_stracks) == 0):
            activated_stracks = []
            for i in range(output_results.shape[0]):
        # Use a reasonable threshold, e.g., 0.1, to avoid tracking noise
                if output_results[i][4] < 0.1:
                    continue
                det = output_results[i]
                track = STrack(STrack.tlbr_to_tlwh(det[:4]), det[4], int(det[5]) if len(det) > 5 else -1)
                track.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(track)
            self.tracked_stracks = activated_stracks
            return self.tracked_stracks
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            if hasattr(output_results, "cpu"):
                output_results = output_results.cpu().numpy()
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, int(c))
              for tlbr, s, c in zip(dets, scores_keep, output_results[remain_inds, 5] if output_results.shape[1] > 5 else [-1]*len(dets))]
        else:
            detections = []

        if isinstance(detections, np.ndarray):
            detections = [
                STrack(
                    STrack.tlbr_to_tlwh(det[:4]), 
                    det[4], 
                    int(det[5]) if len(det) > 5 else -1
                ) for det in detections
            ]
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        print("DEBUG: matches:", matches)
        print("DEBUG: u_track (unmatched tracks):", u_track)
        print("DEBUG: u_detection (unmatched detections):", u_detection)
        print("DEBUG: detections:", detections)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=True)
                refind_stracks.append(track)

        if len(detections) > 0 and not isinstance(detections[0], STrack):
            detections = [
                STrack(
                    STrack.tlbr_to_tlwh(np.asarray(det[:4], dtype=np.float32)),
                    float(det[4]),
                    int(det[5]) if len(det) > 5 else -1
                ) for det in detections
            ]
        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
            #track.mark_removed()
            #removed_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, int(c))
                    for (tlbr, s, c) in zip(dets_second, scores_second, output_results[inds_second, 5] if output_results.shape[1] > 5 else [-1]*len(dets_second))]

        else:
            detections_second = []
        
        if len(detections_second) > 0 and not isinstance(detections_second[0], STrack):
            detections_second = [
                STrack(
                    STrack.tlbr_to_tlwh(np.asarray(det[:4], dtype=np.float32)),
                    float(det[4]),
                    int(det[5]) if len(det) > 5 else -1
                )
                for det in detections_second
            ]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=True)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            #if not track.state == TrackState.Lost:
                #track.mark_lost()
                #lost_stracks.append(track)
            track.mark_removed()
            removed_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        if len(detections) > 0 and not isinstance(detections[0], STrack):
            detections = [
                STrack(
                    STrack.tlbr_to_tlwh(np.asarray(det[:4], dtype=np.float32)),
                    float(det[4]),
                    int(det[5]) if len(det) > 5 else -1
                )
                for det in detections
            ]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        # Step 4: Hybrid association for unmatched detections
        all_det_indices = set(range(len(detections)))
        matched_det_indices = set([idet for _, idet in matches])
        unmatched_det_indices = all_det_indices - matched_det_indices

        for inew in unmatched_det_indices:
            track = detections[inew]
            if track.score < 0.1:
                continue

            # Try to associate with a lost/existing track by distance
            closest_track = None
            min_distance = float('inf')
            for existing_track in self.tracked_stracks + self.lost_stracks:
                existing_center = np.array(existing_track.tlwh[:2]) + np.array(existing_track.tlwh[2:]) / 2
                new_center = np.array(track.tlwh[:2]) + np.array(track.tlwh[2:]) / 2
                distance = np.linalg.norm(existing_center - new_center)
                if distance < min_distance:
                    min_distance = distance
                    closest_track = existing_track

            if closest_track is not None and min_distance < 75:
                print(f"DEBUG: Forcing update of track {closest_track.track_id} with detection (min_dist={min_distance:.1f})")
                closest_track.update(track, self.frame_id)
                activated_starcks.append(closest_track)
            else:
                print(f"DEBUG: Creating new track for distant detection (min_dist={min_distance:.1f}):", track)
                new_track = STrack(track.tlwh, track.score, track.class_id)
                new_track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(new_track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]

        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        #output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        #output_stracks = [track for track in self.tracked_stracks if track.is_activated] + \
                         #[track for track in refind_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks if track.is_activated and track.state == TrackState.Tracked] + \
                 [track for track in refind_stracks if track.is_activated and track.state == TrackState.Tracked]
        print("BYTETracker output_stracks:", output_stracks)
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.9)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
