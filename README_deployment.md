# Pose Animation Analyzer

A Streamlit web application for analyzing body pose movements from video files using MediaPipe.

## Features

- **Pose Detection**: Extract 33 body landmarks from video files
- **Animated Skeleton**: Interactive skeleton visualization
- **Trajectory Analysis**: Movement path analysis of key body parts
- **Joint Analysis**: Sport-specific biomechanical metrics (Archery)
- **AI Analysis**: OpenAI-powered biomechanics reports
- **Statistics**: Comprehensive pose data analysis

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run pose_app.py
```

## Deployment Options

### 1. Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository and main file (`pose_app.py`)
5. Deploy!

### 2. Heroku

1. Create a `Procfile`:
```
web: streamlit run pose_app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy using Heroku CLI or GitHub integration

### 3. Railway

1. Connect your GitHub repository to Railway
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run pose_app.py --server.port=$PORT`

### 4. Docker

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "pose_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build and run:
```bash
docker build -t pose-analyzer .
docker run -p 8501:8501 pose-analyzer
```

## Environment Variables

For AI Analysis feature, set your OpenAI API key:
- `OPENAI_API_KEY`: Your OpenAI API key

## File Structure

- `pose_app.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- `.streamlit/config.toml`: Streamlit configuration
- `output/`: Generated output files
- `README.md`: This file

## Usage

1. Upload a video file (MP4, AVI, MOV, MKV)
2. Adjust detection settings in the sidebar
3. View results across multiple tabs:
   - **Detections**: Annotated video with pose keypoints
   - **Animation**: Interactive skeleton visualization
   - **Trajectories**: Movement path analysis
   - **Statistics**: Pose data metrics
   - **Joint Analysis**: Sport-specific biomechanics
   - **AI Analysis**: OpenAI-powered reports

## Tips for Best Results

- Good lighting conditions
- Clear view of the person
- Appropriate camera distance
- Stable camera position
- Single person in frame 