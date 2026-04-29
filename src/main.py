import cv2
from court_detection_net import CourtDetectorNet
import numpy as np
from court_reference import CourtReference
from bounce_detector import BounceDetector
from person_detector import PersonDetector
from ball_detector import BallDetector
from utils import scene_detect
import argparse
import torch
import xml.etree.ElementTree as ET
from scipy.spatial import distance

# NEW IMPORT
from score_detector import ScoreDetector


# --- MANUAL CALCULATION FUNCTION (USES HARDCODED STATIC PIXELS) ---
def calculate_manual_coords(x_input, y_input, static_kps_data):

    SINGLES_HALF_WIDTH = 4.115 
    COURT_HALF_DEPTH = 11.885
    
    x5 = 423.68
    y5 = 2933.81
    x7 = 1242.15
    y4 = 561.00
    
    x_prime = None
    if x7 - x5 != 0:
        pixel_ratio_x = (x_input - x5) / (x7 - x5)
        x_prime = -SINGLES_HALF_WIDTH + (pixel_ratio_x * (2 * SINGLES_HALF_WIDTH))

    y_prime = None
    if y4 - y5 != 0:
        pixel_ratio_y = (y_input - y5) / (y4 - y5)
        y_prime = -COURT_HALF_DEPTH + (pixel_ratio_y * (2 * COURT_HALF_DEPTH))
        
    return x_prime, y_prime



def smooth_homography_matrices(homography_matrices, window_size=15):
    H_smoothed = [None] * len(homography_matrices)
    
    for i in range(len(homography_matrices)):
        H_window = []
        start_index = max(0, i - window_size // 2)
        end_index = min(len(homography_matrices), i + window_size // 2 + 1)
        
        for j in range(start_index, end_index):
            if homography_matrices[j] is not None:
                H_window.append(homography_matrices[j])
        
        if H_window:
            H_stack = np.stack(H_window, axis=0)
            H_median = np.median(H_stack, axis=0)
            H_smoothed[i] = H_median
        else:
            H_smoothed[i] = homography_matrices[i]

    return H_smoothed



def interpolate_ball_track(ball_track, max_gap=10):
    interpolated_track = list(ball_track)
    start_gap_index = None

    for i in range(len(interpolated_track)):
        if interpolated_track[i][0] is None:
            if start_gap_index is None:
                start_gap_index = i
        else:
            if start_gap_index is not None:
                end_gap_index = i
                gap_length = end_gap_index - start_gap_index
                
                if gap_length <= max_gap and start_gap_index > 0:
                    x_start, y_start = interpolated_track[start_gap_index - 1]
                    x_end, y_end = interpolated_track[end_gap_index]
                    
                    for k in range(gap_length):
                        t = (k + 1) / (gap_length + 1)
                        x_interp = x_start + t * (x_end - x_start)
                        y_interp = y_start + t * (y_end - y_start)
                        interpolated_track[start_gap_index + k] = (x_interp, y_interp)
                
                start_gap_index = None
    
    return interpolated_track



def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break    
    cap.release()
    return frames, fps

def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2)*255).astype(np.uint8)
    return court_img



def get_static_keypoints_data(kps_court, homography_matrices):
    for i in range(len(kps_court)):
        if kps_court[i] is not None and homography_matrices[i] is not None:
            inv_mat = homography_matrices[i]
            keypoint_data = []

            kps_np = np.array(kps_court[i], dtype=np.float32).reshape(-1, 1, 2)
            minimap_kps = cv2.perspectiveTransform(kps_np, inv_mat).squeeze()
            
            for j in range(len(kps_court[i])):
                if j < len(minimap_kps):
                    keypoint_data.append({
                        'x': float(minimap_kps[j][0]),
                        'y': float(minimap_kps[j][1])
                    })
            return keypoint_data

    return []



def write_bounces_to_xml(bounces_data, court_keypoints_data, output_path, fps):

    root = ET.Element('Bounces')
    
    if court_keypoints_data:
        court_data_elem = ET.SubElement(root, 'StaticCourtData')
        court_kps_elem = ET.SubElement(court_data_elem, 'CourtKeypoints')
        
        for kp_index, kp in enumerate(court_keypoints_data):
            ET.SubElement(court_kps_elem, 'Keypoint',
                          id=str(kp_index),
                          x=str(round(kp['x'], 2)),
                          y=str(round(kp['y'], 2)))


    for hit_id, bounce in enumerate(bounces_data):

        time_seconds = bounce['frame_number'] / fps
        
        bounce_elem = ET.SubElement(root, 'hit', 
                                    id=str(hit_id + 1),
                                    frame_number=str(bounce['frame_number']),
                                    time_seconds=str(round(time_seconds, 2)))
        
        if bounce['ball_x'] is not None and bounce['ball_y'] is not None:
            ET.SubElement(bounce_elem, 'HomographyMeterCoordinates',
                          x_meter=str(round(bounce['ball_x'], 3)),
                          y_meter=str(round(bounce['ball_y'], 3)))

        if bounce['manual_x'] is not None and bounce['manual_y'] is not None:
            ET.SubElement(bounce_elem, 'ManualMeterCoordinates',
                          x_prime=str(round(bounce['manual_x'], 3)),
                          y_prime=str(round(bounce['manual_y'], 3)))

        players_elem = ET.SubElement(bounce_elem, 'Players')

        for player in bounce['players']:
            player_elem = ET.SubElement(players_elem, 'Player')
            
            if player['homography_x'] is not None and player['homography_y'] is not None:
                ET.SubElement(player_elem, 'HomographyPlayerCoordinates',
                              x_meter=str(round(player['homography_x'], 3)),
                              y_meter=str(round(player['homography_y'], 3)))

            if player['manual_x'] is not None and player['manual_y'] is not None:
                ET.SubElement(player_elem, 'ManualPlayerCoordinates',
                              x_prime=str(round(player['manual_x'], 3)),
                              y_prime=str(round(player['manual_y'], 3)))


        # NEW: SCORE SECTION
        ET.SubElement(
            bounce_elem,
            "Score",
            player_top=str(bounce.get("player_top")),
            player_bottom=str(bounce.get("player_bottom")),
            games_top=str(bounce.get("games_top")),
            points_top=str(bounce.get("points_top")),
            games_bottom=str(bounce.get("games_bottom")),
            points_bottom=str(bounce.get("points_bottom")),
        )

    tree = ET.ElementTree(root)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)



def main(frames, scenes, bounces, ball_track, homography_matrices, kps_court,
         persons_top, persons_bottom, court_keypoints_data, draw_trace=False, trace=7):

    imgs_res = []
    bounces_data = []

    width_minimap = 166
    height_minimap = 350

    is_track = [x is not None for x in homography_matrices]

    # NEW: store last OCR score
    last_score = {
        "player_top": None,
        "player_bottom": None,
        "games_top": None,
        "points_top": None,
        "games_bottom": None,
        "points_bottom": None
    }

    # Score detector instance
    score_detector = ScoreDetector()

    
    for num_scene in range(len(scenes)):

        sum_track = sum(is_track[scenes[num_scene][0]:scenes[num_scene][1]])
        len_track = scenes[num_scene][1] - scenes[num_scene][0]

        scene_rate = sum_track / (len_track + 1e-15)

        if scene_rate > 0.5:
            court_img = get_court_img()

            for i in range(scenes[num_scene][0], scenes[num_scene][1]):

                img_res = frames[i]
                inv_mat = homography_matrices[i]

                # --- READ SCOREBOARD ONCE PER SECOND ---
                if fps > 0 and i % fps == 0:
                    current_score = score_detector.read_score(img_res)
                    if current_score["player_top"] and current_score["player_bottom"]:
                        last_score = current_score


                player_data = [] 
                
                ball_x_pixel, ball_y_pixel = None, None
                homography_ball_x, homography_ball_y = None, None 
                
                if ball_track[i][0] is not None:
                    ball_x_pixel = ball_track[i][0]
                    ball_y_pixel = ball_track[i][1]

                    if not draw_trace:
                        img_res = cv2.circle(img_res, (int(ball_x_pixel), int(ball_y_pixel)),
                                             radius=5, color=(0, 255, 0), thickness=2)
                    
                    if inv_mat is not None:
                        ball_point_np = np.array(ball_track[i], dtype=np.float32).reshape(1, 1, 2)
                        minimap_ball_coords = cv2.perspectiveTransform(ball_point_np, inv_mat)
                        
                        homography_ball_x = float(minimap_ball_coords[0, 0, 0])
                        homography_ball_y = float(minimap_ball_coords[0, 0, 1])


                if kps_court[i] is not None and inv_mat is not None:

                    for j in range(len(kps_court[i])):
                        try:
                            img_res = cv2.circle(
                                img_res,
                                (int(kps_court[i][j][0, 0]), int(kps_court[i][j][0, 1])),
                                radius=0, color=(0, 0, 255), thickness=10)
                        except:
                            pass
                        
                manual_x, manual_y = None, None

                if i in bounces and inv_mat is not None:
                    if homography_ball_x is not None:
                        court_img = cv2.circle(
                            court_img,
                            (int(homography_ball_x), int(homography_ball_y)),
                            radius=0, color=(0, 255, 255), thickness=50)

                    if court_keypoints_data and homography_ball_x is not None:
                        manual_x, manual_y = calculate_manual_coords(
                            homography_ball_x, homography_ball_y, court_keypoints_data
                        )
                    
                minimap = court_img.copy()

                persons = persons_top[i] + persons_bottom[i]                    

                for j, person in enumerate(persons):
                    if len(person[0]) > 0:
                        person_point_np = np.array(person[1], dtype=np.float32).reshape(1, 1, 2)
                        minimap_person_coords = cv2.perspectiveTransform(person_point_np, inv_mat)
                        
                        homography_player_x = float(minimap_person_coords[0, 0, 0])
                        homography_player_y = float(minimap_person_coords[0, 0, 1])

                        minimap = cv2.circle(
                            minimap,
                            (int(homography_player_x), int(homography_player_y)),
                            radius=0, color=(255, 0, 0), thickness=80)

                        p_manual_x, p_manual_y = calculate_manual_coords(
                            homography_player_x, homography_player_y, court_keypoints_data
                        )
                        
                        player_data.append({
                            'homography_x': homography_player_x,
                            'homography_y': homography_player_y,
                            'manual_x': p_manual_x,
                            'manual_y': p_manual_y
                        })


                minimap = cv2.resize(minimap, (width_minimap, height_minimap))
                h, w, _ = img_res.shape
                img_res[30:(30 + height_minimap), (w - 30 - width_minimap):(w - 30), :] = minimap

                imgs_res.append(img_res)
                
                if i in bounces:
                    bounce_info = {
                        'frame_number': i,
                        'ball_x': homography_ball_x, 
                        'ball_y': homography_ball_y,
                        'manual_x': manual_x, 
                        'manual_y': manual_y,
                        'players': player_data,

                        # NEW SCORE FIELDS
                        'player_top': last_score["player_top"],
                        'player_bottom': last_score["player_bottom"],
                        'games_top': last_score["games_top"],
                        'points_top': last_score["points_top"],
                        'games_bottom': last_score["games_bottom"],
                        'points_bottom': last_score["points_bottom"],
                    }

                    bounces_data.append(bounce_info)

        else:
            imgs_res += frames[scenes[num_scene][0]:scenes[num_scene][1]]

    return imgs_res, bounces_data



def write(imgs_res, fps, path_output_video):
    height, width = imgs_res[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for frame in imgs_res:
        out.write(frame)
    out.release()    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_ball_track_model', type=str)
    parser.add_argument('--path_court_model', type=str)
    parser.add_argument('--path_bounce_model', type=str)
    parser.add_argument('--path_input_video', type=str)
    parser.add_argument('--path_output_video', type=str)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    frames, fps = read_video(args.path_input_video)
    scenes = scene_detect(args.path_input_video)

    print('ball detection')
    ball_detector = BallDetector(args.path_ball_track_model, device)
    ball_track = ball_detector.infer_model(frames)

    ball_track = interpolate_ball_track(ball_track, max_gap=10)

    print('court detection')
    court_detector = CourtDetectorNet(args.path_court_model, device)
    homography_matrices, kps_court = court_detector.infer_model(frames)
    homography_matrices = smooth_homography_matrices(homography_matrices, window_size=15)

    print('person detection')
    person_detector = PersonDetector(device)
    persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=False)

    bounce_detector = BounceDetector(args.path_bounce_model)
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    court_keypoints_data = get_static_keypoints_data(kps_court, homography_matrices)

    imgs_res, bounces_data = main(
        frames, scenes, bounces, ball_track,
        homography_matrices, kps_court,
        persons_top, persons_bottom,
        court_keypoints_data,
        draw_trace=True
    )

    write(imgs_res, fps, args.path_output_video)
    write_bounces_to_xml(bounces_data, court_keypoints_data, 'bounces.xml', fps)
