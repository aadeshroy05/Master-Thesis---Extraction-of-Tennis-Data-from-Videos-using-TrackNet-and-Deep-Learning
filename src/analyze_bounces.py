import xml.etree.ElementTree as ET
from scipy.spatial import distance
import numpy as np

# Gravity constant in m/s^2 (used for vertical trajectory calculations)
G = 9.81 

# --- ZONE CLASSIFICATION FUNCTION ---
def classify_shot_zone(x_m, y_m):
    """
    Classifies a bounce location into a strategic court zone (in meters).
    This uses the known court boundaries (Singles: +/-4.115m, Depth: +/-11.885m)
    and Service Line depth (+/-6.40m).
    
    Returns: A strategic zone label.
    """
    SINGLES_HALF_WIDTH = 4.115
    SERVICE_LINE_Y = 6.40
    BASELINE_Y = 11.885
    
    # 1. Check Bounds (Must be within the Singles Court for analysis)
    if abs(x_m) > SINGLES_HALF_WIDTH or abs(y_m) > BASELINE_Y:
        return "OUT_OF_BOUNDS"

    # 2. Classify by Depth (Y-axis)
    
    # Opponent's Side (Y is Positive)
    if y_m > 0:
        if y_m <= SERVICE_LINE_Y:
            return "OPPONENT_SERVICE_BOX"
        elif y_m <= BASELINE_Y:
            return "OPPONENT_DEEP_BASELINE"
            
    # Your Side (Y is Negative)
    elif y_m < 0:
        if abs(y_m) <= SERVICE_LINE_Y:
            return "YOUR_SERVICE_BOX"
        elif abs(y_m) <= BASELINE_Y:
            return "YOUR_DEEP_BASELINE"

    # Edge cases (Net, Center Line boundaries)
    return "UNKNOWN_NET_REGION"


def estimate_missing_bounce(prev_bounce, next_bounce):
    """
    Estimates a missing bounce using a simple parabolic trajectory model.
    This function is a placeholder and assumes vertical velocity is zero at peak.
    """
    
    # 1. Calculate time difference
    time_s = next_bounce['landing_time_s'] - prev_bounce['landing_time_s']
    
    # 2. Estimate peak height and duration (assuming symmetry)
    # Time to peak (T/2) = (time_s / 2)
    # Height (H) = 0.5 * G * (T/2)^2 
    time_to_peak = time_s / 2
    height_m = 0.5 * G * (time_to_peak ** 2)
    
    # 3. Estimate landing coordinates (midpoint of the two known bounces)
    mid_x = (prev_bounce['landing_x_m'] + next_bounce['landing_x_m']) / 2
    mid_y = (prev_bounce['landing_y_m'] + next_bounce['landing_y_m']) / 2
    mid_time = prev_bounce['landing_time_s'] + time_to_peak
    
    # 4. Create a placeholder frame number (negative to signal it's an estimation)
    # Note: A proper implementation would use the raw ball_track data to find the actual peak pixel.
    
    estimated_bounce = {
        'bounce_frame': f"EST_{int(mid_time * 30)}", # Placeholder for estimated frame based on time (assuming 30fps)
        'landing_x_m': round(mid_x, 3),
        'landing_y_m': round(mid_y, 3),
        'landing_time_s': round(mid_time, 3),
        'is_estimated': True,
        'estimated_peak_height_m': round(height_m, 3)
    }
    return estimated_bounce


def analyze_shot_metrics(xml_filepath, max_flight_time=0.9):
    """
    Loads bounce data from XML, calculates metrics, and attempts to correct 
    missing bounces using physics-based estimation if flight time exceeds max_flight_time.
    """
    try:
        tree = ET.parse(xml_filepath)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_filepath}")
        return []

    bounces = root.findall('Bounce')
    if not bounces:
        print("No bounce data found in XML.")
        return []

    processed_bounces = []
    
    # Store the coordinates of the previous bounce to calculate the current shot
    prev_bounce_data = None 

    for i in range(len(bounces)):
        current_bounce_xml = bounces[i]
        
        # Extract Meter Coordinates
        coords_elem = current_bounce_xml.find('HomographyMeterCoordinates')
        time_str = current_bounce_xml.get('time_seconds')

        if coords_elem is None or time_str is None:
             continue # Skip if essential data is missing
        
        curr_x = float(coords_elem.get('x_meter'))
        curr_y = float(coords_elem.get('y_meter'))
        curr_time = float(time_str)

        current_bounce_data = {
            'bounce_frame': current_bounce_xml.get('frame_number'),
            'landing_x_m': curr_x,
            'landing_y_m': curr_y,
            'landing_time_s': curr_time,
            'flight_distance_m': None,
            'flight_time_s': None,
            'average_speed_mps': None,
            'is_estimated': False, # Confirmed bounce
            'zone': classify_shot_zone(curr_x, curr_y) # CLASSIFY ZONE
        }

        if prev_bounce_data is not None:
            # --- METRICS CALCULATION ---
            prev_x = prev_bounce_data['landing_x_m']
            prev_y = prev_bounce_data['landing_y_m']
            prev_time = prev_bounce_data['landing_time_s']

            dist_m = distance.euclidean([prev_x, prev_y], [curr_x, curr_y])
            time_s = curr_time - prev_time
            speed_mps = dist_m / time_s if time_s > 1e-6 else 0
            
            # Store metrics in current bounce
            current_bounce_data['flight_distance_m'] = round(dist_m, 3)
            current_bounce_data['flight_time_s'] = round(time_s, 3)
            current_bounce_data['average_speed_mps'] = round(speed_mps, 3)

            # --- CORRECTION LOGIC (PHYSICS-BASED ESTIMATION) ---
            # If flight time is too long, it suggests a missed bounce in the middle.
            if time_s > max_flight_time:
                # Insert a new estimated bounce between prev_bounce_data and current_bounce_data
                estimated_bounce = estimate_missing_bounce(prev_bounce_data, current_bounce_data)
                
                # Append the estimated bounce first
                processed_bounces.append(estimated_bounce)
                
                # Recalculate metrics for the newly split segments
                # A. Segment 1: Prev to Estimated
                time_s1 = estimated_bounce['landing_time_s'] - prev_time
                dist_m1 = distance.euclidean([prev_x, prev_y], [estimated_bounce['landing_x_m'], estimated_bounce['landing_y_m']])
                speed_mps1 = dist_m1 / time_s1 if time_s1 > 1e-6 else 0
                
                # Reclassify the zone for the estimated bounce
                estimated_bounce['zone'] = classify_shot_zone(estimated_bounce['landing_x_m'], estimated_bounce['landing_y_m'])
                
                processed_bounces[-1]['flight_distance_m'] = round(dist_m1, 3)
                processed_bounces[-1]['flight_time_s'] = round(time_s1, 3)
                processed_bounces[-1]['average_speed_mps'] = round(speed_mps1, 3)

                # B. Segment 2: Estimated to Current (Recalculate metrics for the current bounce entry)
                time_s2 = curr_time - estimated_bounce['landing_time_s']
                dist_m2 = distance.euclidean([estimated_bounce['landing_x_m'], estimated_bounce['landing_y_m']], [curr_x, curr_y])
                speed_mps2 = dist_m2 / time_s2 if time_s2 > 1e-6 else 0

                current_bounce_data['flight_distance_m'] = round(dist_m2, 3)
                current_bounce_data['flight_time_s'] = round(time_s2, 3)
                current_bounce_data['average_speed_mps'] = round(speed_mps2, 3)
        
        processed_bounces.append(current_bounce_data)
        
        # Set current bounce as the previous bounce for the next iteration
        prev_bounce_data = current_bounce_data

    return processed_bounces

if __name__ == '__main__':
    # You must run main.py first to generate a valid bounces.xml file!
    
    # We set a max flight time (0.9s) as a threshold. If a shot takes longer, 
    # the model likely missed the bounce on the other side of the net.
    results = analyze_shot_metrics('bounces.xml', max_flight_time=0.9)
    
    if results:
        # Calculate summary statistics
        zone_counts = {}
        speed_sums = {}
        total_shots = 0
        
        for shot in results[1:]:
            zone = shot['zone']
            speed = shot['average_speed_mps']
            
            # Count shots per zone
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
            
            # Sum speeds for average calculation
            speed_sums[zone] = speed_sums.get(zone, (0, 0))
            if speed is not None:
                 # Tuple structure: (Total Speed Sum, Total Shot Count)
                 speed_sums[zone] = (speed_sums[zone][0] + speed, speed_sums[zone][1] + 1)
            
            total_shots += 1
            
        print("--- FINAL STRATEGIC ANALYSIS REPORT ---")
        print(f"Total Confirmed/Estimated Shots Analyzed: {total_shots}")
        print("-" * 40)
        print("SHOT COUNTS BY STRATEGIC ZONE:")
        
        # Sort and print results
        sorted_zones = sorted(zone_counts.items(), key=lambda item: item[1], reverse=True)
        
        for zone, shot_count in sorted_zones:
            total_speed, speed_count = speed_sums[zone]
            avg_speed = (total_speed / speed_count) if speed_count > 0 else 0
            
            print(f"  {zone.replace('_', ' ')}:")
            print(f"    - Count: {shot_count} ({shot_count/total_shots:.1%})")
            print(f"    - Avg Speed: {avg_speed:.2f} m/s")
        
        print("-" * 40)
        print("Insight: High counts in OPPONENT_DEEP_BASELINE indicate a defensive rally.")
