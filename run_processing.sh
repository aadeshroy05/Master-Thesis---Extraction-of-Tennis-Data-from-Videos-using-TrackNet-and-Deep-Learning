#!/bin/bash

# --- CONFIGURATION ---
TOTAL_CLIPS=41
INPUT_DIR="inputvideos_splited"
OUTPUT_DIR="Overlap_output_videos"
PYTHON_SCRIPT="python3 main.py"
MODEL_ARGS="--path_ball_track_model tracknet_model.pt --path_court_model court_model.pt --path_bounce_model bounce_model.cbm"

# 1. Ensure the output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

echo "Starting batch processing of $TOTAL_CLIPS video clips..."

# 2. Loop through the clip numbers (1 to 41)
for i in $(seq 1 $TOTAL_CLIPS); do
    
    # Define the FULL paths for current clip
    INPUT_FILE_NAME="output${i}.mp4"
    INPUT_FILE_PATH="$INPUT_DIR/$INPUT_FILE_NAME"
    
    OUTPUT_VIDEO_FILE="$OUTPUT_DIR/final_output${i}.mp4"
    OUTPUT_XML_FILE="$OUTPUT_DIR/bounces_${i}.xml"  # XML files go in the output folder too
    
    echo "--- Processing $INPUT_FILE_PATH (Clip $i/$TOTAL_CLIPS) ---"

    # 3. Execute the Python Command
    # The command uses the input path and the output video path
    $PYTHON_SCRIPT \
        $MODEL_ARGS \
        --path_input_video "$INPUT_FILE_PATH" \
        --path_output_video "$OUTPUT_VIDEO_FILE"

    # 4. RENAME AND MOVE THE XML FILE
    # The Python script creates 'bounces.xml' in the current working directory.
    # We move it to the OUTPUT_DIR with a unique name.
    if [ -f "bounces.xml" ]; then
        mv bounces.xml "$OUTPUT_XML_FILE"
        echo "Saved XML to $OUTPUT_XML_FILE"
    else
        echo "Warning: bounces.xml not found for $INPUT_FILE_NAME"
    fi

done

echo "Processing complete. Results are in the $OUTPUT_DIR folder."
