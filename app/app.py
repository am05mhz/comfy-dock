#!/usr/bin/env python3
"""
ComfyUI Browser Interface
Fixed implementation with proper workflow execution and multi-workflow support
"""

import os
import json
import uuid
import time
import base64
import logging
import threading
import hashlib

# Check for correct websocket package
try:
    import websocket
    if not hasattr(websocket, 'WebSocketApp'):
        print("\nERROR: Wrong websocket package installed!")
        print("Please run:")
        print("  pip uninstall websocket")
        print("  pip install websocket-client==1.7.0")
        print("\nOr install all requirements:")
        print("  pip install -r requirements.txt\n")
        exit(1)
except ImportError:
    print("\nERROR: websocket-client not installed!")
    print("Please run: pip install websocket-client==1.7.0\n")
    exit(1)

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from queue import Queue, Empty

import requests
from PIL import Image
from flask import Flask, render_template_string, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Configuration
class Config:
    COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
    COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
    PORT = int(os.getenv("PORT", "5000"))
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "outputs")
    WORKFLOWS_FOLDER = os.getenv("WORKFLOWS_FOLDER", "workflows")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def comfyui_url(self):
        return f"http://{self.COMFYUI_HOST}:{self.COMFYUI_PORT}"
    
    @property
    def comfyui_ws_url(self):
        return f"ws://{self.COMFYUI_HOST}:{self.COMFYUI_PORT}"

config = Config()

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directories
for folder in [config.UPLOAD_FOLDER, config.OUTPUT_FOLDER, config.WORKFLOWS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

@dataclass
class GenerationJob:
    id: str
    workflow: Dict[str, Any]
    status: str = "queued"
    progress: float = 0.0
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    current_node: Optional[str] = None

class ComfyUIWebSocketClient:
    """WebSocket client for ComfyUI real-time updates"""
    
    def __init__(self, server_address: str):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.ws = None
        self.messages = Queue()
        self.jobs = {}  # Track jobs by prompt_id
        self.running = False
        self.thread = None
        
    def connect(self):
        """Connect to ComfyUI WebSocket"""
        ws_url = f"ws://{self.server_address}/ws?clientId={self.client_id}"
        
        def on_message(ws, message):
            try:
                # ComfyUI sends both text and binary messages
                if isinstance(message, bytes):
                    # Binary messages are usually image previews, skip them
                    return
                    
                data = json.loads(message)
                self.process_message(data)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse WebSocket message: {message[:100]}...")
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            # Don't let encoding errors kill the connection
            if "codec" not in str(error):
                self.running = False
            
        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")
            self.running = False
            
        def on_open(ws):
            logger.info("WebSocket connection established")
            self.running = True
            
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_data=None  # Let on_message handle both text and binary
        )
        
        self.thread = threading.Thread(target=self.ws.run_forever)
        self.thread.daemon = True
        self.thread.start()
        time.sleep(1)  # Give time to connect
        
    def process_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket message"""
        msg_type = data.get('type')
        msg_data = data.get('data', {})
        
        if msg_type == 'status':
            # Queue status update
            queue_info = msg_data.get('status', {}).get('exec_info', {})
            logger.debug(f"Queue status: {queue_info}")
            
        elif msg_type == 'execution_start':
            prompt_id = msg_data.get('prompt_id')
            if prompt_id and prompt_id in self.jobs:
                self.jobs[prompt_id].status = 'running'
                logger.info(f"Execution started: {prompt_id}")
                
        elif msg_type == 'executing':
            prompt_id = msg_data.get('prompt_id')
            node = msg_data.get('node')
            if prompt_id and prompt_id in self.jobs:
                self.jobs[prompt_id].current_node = node
                logger.info(f"Executing node {node} for prompt {prompt_id}")
                
        elif msg_type == 'progress':
            prompt_id = msg_data.get('prompt_id')
            if prompt_id and prompt_id in self.jobs:
                value = msg_data.get('value', 0)
                max_value = msg_data.get('max', 100)
                self.jobs[prompt_id].progress = (value / max_value) if max_value > 0 else 0
                logger.debug(f"Progress for {prompt_id}: {value}/{max_value}")
                
        elif msg_type == 'executed':
            prompt_id = msg_data.get('prompt_id')
            node = msg_data.get('node')
            output = msg_data.get('output', {})
            logger.info(f"Node {node} executed for prompt {prompt_id}")
            
        elif msg_type == 'execution_error':
            prompt_id = msg_data.get('prompt_id')
            if prompt_id and prompt_id in self.jobs:
                error = msg_data.get('exception_message', 'Unknown error')
                self.jobs[prompt_id].status = 'failed'
                self.jobs[prompt_id].error = error
                logger.error(f"Execution error for {prompt_id}: {error}")
                
        elif msg_type == 'execution_cached':
            logger.debug(f"Cached execution: {msg_data}")

class ComfyUIClient:
    """Client for interacting with ComfyUI API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.ws_client = ComfyUIWebSocketClient(base_url.replace('http://', ''))
        self.ws_client.connect()
        
    def queue_prompt(self, prompt: Dict[str, Any]) -> str:
        """Queue a prompt in ComfyUI"""
        p = {"prompt": prompt, "client_id": self.ws_client.client_id}
        
        try:
            response = requests.post(f"{self.base_url}/prompt", json=p)
            response.raise_for_status()
            result = response.json()
            prompt_id = result['prompt_id']
            
            # Create job tracking
            job = GenerationJob(id=prompt_id, workflow=prompt)
            self.ws_client.jobs[prompt_id] = job
            
            logger.info(f"Queued prompt: {prompt_id}")
            return prompt_id
            
        except Exception as e:
            logger.error(f"Failed to queue prompt: {e}")
            raise
            
    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get execution history for a prompt"""
        try:
            response = requests.get(f"{self.base_url}/history/{prompt_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return {}
            
    def get_queue_status(self) -> Dict[str, Any]:
        """Get ComfyUI queue status"""
        try:
            response = requests.get(f"{self.base_url}/queue")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {}
            
    def interrupt_execution(self) -> bool:
        """Interrupt current execution"""
        try:
            response = requests.post(f"{self.base_url}/interrupt")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to interrupt execution: {e}")
            return False
        
    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download an image from ComfyUI"""
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        
        try:
            response = requests.get(f"{self.base_url}/view", params=params)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to get image {filename}: {e}")
            return None            
        
    def get_queue_status(self) -> Dict[str, Any]:
        """Get ComfyUI queue status"""
        try:
            response = requests.get(f"{self.base_url}/queue")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {}

    def interrupt_execution(self) -> bool:
            """Interrupt current execution"""
            try:
                response = requests.post(f"{self.base_url}/interrupt")
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Failed to interrupt execution: {e}")
                return False
            
    def upload_image(self, filepath: str, name: Optional[str] = None, overwrite: bool = True) -> str:
        """Upload an image to ComfyUI"""
        if name is None:
            name = os.path.basename(filepath)
            
        with open(filepath, 'rb') as f:
            files = {'image': (name, f, 'image/png')}
            data = {'overwrite': str(overwrite).lower()}
            
            try:
                response = requests.post(
                    f"{self.base_url}/upload/image",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                result = response.json()
                return result.get('name', name)
            except Exception as e:
                logger.error(f"Failed to upload image: {e}")
                raise
                
    def wait_for_job(self, prompt_id: str, timeout: int = 300) -> GenerationJob:
        """Wait for a job to complete"""
        start_time = time.time()
        job = self.ws_client.jobs.get(prompt_id)
        
        if not job:
            # Create a new job if not tracked
            job = GenerationJob(id=prompt_id, workflow={})
            self.ws_client.jobs[prompt_id] = job
            
        last_status = None
        no_change_count = 0
        
        while time.time() - start_time < timeout:
            # Check history for completion
            history = self.get_history(prompt_id)
            
            if prompt_id in history:
                hist_data = history[prompt_id]
                outputs = hist_data.get('outputs', {})
                
                # Log status changes
                current_status = f"{job.status}:{len(outputs)}"
                if current_status != last_status:
                    logger.info(f"Job {prompt_id} status: {job.status}, outputs: {len(outputs)}")
                    last_status = current_status
                    no_change_count = 0
                else:
                    no_change_count += 1
                    
                # If no change for too long, check if it's actually complete
                if no_change_count > 20 and outputs:  # 10 seconds of no change
                    logger.warning(f"Job {prompt_id} appears stuck but has outputs, forcing completion")
                    
                # Check if execution is complete by looking for status in history
                if 'status' in hist_data and hist_data['status'].get('status_str') == 'success':
                    logger.info(f"Job {prompt_id} completed successfully")
                
                # Check if we have all expected outputs
                if outputs:
                    # Log what outputs we found
                    for node_id, node_output in outputs.items():
                        if 'images' in node_output:
                            logger.info(f"Found {len(node_output['images'])} images from node {node_id}")
                    
                    # Collect all output images
                    # Collect all output images
                    for node_id, node_output in outputs.items():
                        if 'images' in node_output:
                            for img in node_output['images']:
                                # Check if we already have this output
                                existing = any(o['node_id'] == node_id and 
                                             o.get('original_filename') == img['filename'] 
                                             for o in job.outputs)
                                if existing:
                                    continue
                                    
                                # Download the image
                                img_data = self.get_image(
                                    img['filename'],
                                    img.get('subfolder', ''),
                                    img.get('type', 'output')
                                )
                                
                                if img_data:
                                    # Save image locally
                                    output_filename = f"{prompt_id}_{img['filename']}"
                                    output_path = os.path.join(config.OUTPUT_FOLDER, output_filename)
                                    
                                    with open(output_path, 'wb') as f:
                                        f.write(img_data)
                                        
                                    job.outputs.append({
                                        'filename': output_filename,
                                        'node_id': node_id,
                                        'path': output_path,
                                        'original_filename': img['filename']
                                    })
                                    
                                    logger.info(f"Saved output image: {output_filename}")
                    
                    # Mark as completed if we have outputs
                    if job.outputs:
                        job.status = 'completed'
                        job.completed_at = datetime.now()
                        logger.info(f"Job {prompt_id} completed with {len(job.outputs)} outputs")
                        return job
                    elif no_change_count > 40:  # 20 seconds with outputs but no new ones
                        logger.warning(f"Job {prompt_id} has {len(outputs)} outputs but no new images, marking complete")
                        job.status = 'completed'
                        job.completed_at = datetime.now()
                        return job
                        
            # Check for failure
            if job.status == 'failed':
                raise Exception(f"Job failed: {job.error}")
                
            # Check if workflow might be stuck
            if time.time() - start_time > 30 and job.status == 'queued':
                logger.warning(f"Job {prompt_id} stuck in queued status")
                # Try to get queue info
                try:
                    queue_response = requests.get(f"{self.base_url}/queue")
                    if queue_response.status_code == 200:
                        queue_data = queue_response.json()
                        logger.info(f"Queue status: {json.dumps(queue_data, indent=2)}")
                except:
                    pass
                    
            time.sleep(0.5)
            
        raise TimeoutError(f"Job {prompt_id} timed out after {timeout} seconds")

class WorkflowManager:
    """Manages ComfyUI workflows"""
    
    def __init__(self):
        self.workflows = {}
        self.load_workflows()
        
    def load_workflows(self):
        """Load all workflow JSON files from workflows folder"""
        workflows_path = Path(config.WORKFLOWS_FOLDER)
        
        # Ensure workflows folder exists
        workflows_path.mkdir(exist_ok=True)
        
        for json_file in workflows_path.glob('*.json'):
            try:
                # with open(json_file, 'r') as f:
                #     workflow_data = json.load(f)
                    
                # Check if it's a full ComfyUI workflow format (has 'nodes' and 'links')
                # if 'nodes' in workflow_data and 'links' in workflow_data:
                #     logger.warning(f"{json_file.name} appears to be in full ComfyUI format, not API format")
                #     # You could add conversion logic here if needed
                #     continue
                    
                workflow_id = json_file.stem  # filename without extension
                workflow_name = workflow_id.replace('_', ' ').replace('-', ' ').title()
                
                # Try to extract description from workflow
                description = f"Workflow from {json_file.name}"
                # if '_meta' in workflow_data:
                #     description = workflow_data['_meta'].get('description', description)
                    
                self.workflows[workflow_id] = {
                    'id': workflow_id,
                    'name': workflow_name,
                    # 'workflow': workflow_data,
                    'description': description,
                    'filename': json_file.name
                }
                
                logger.info(f"Loaded workflow: {workflow_name} from {json_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
                
        if len(self.workflows) == 0:
            logger.warning("No workflows loaded! Make sure to:")
            logger.warning("1. Create a 'workflows' folder")
            logger.warning("2. Export workflows from ComfyUI using 'Save (API Format)' button")
            logger.warning("3. Place the JSON files in the workflows folder")
                
        logger.info(f"Total workflows loaded: {len(self.workflows)}")
        
    def analyze_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow and extract user inputs"""
        inputs = {}
        
        for node_id, node in workflow.items():
            # Skip metadata
            if node_id.startswith('_'):
                continue
                
            class_type = node.get('class_type', '')
            node_inputs = node.get('inputs', {})
            meta = node.get('_meta', {})
            title = meta.get('title', class_type)
            
            # LoadImage nodes
            if class_type == 'LoadImage':
                inputs[f"image_{node_id}"] = {
                    'type': 'image',
                    'node_id': node_id,
                    'title': title,
                    'required': True
                }
                
            # Text prompt nodes
            elif class_type == 'CLIPTextEncode':
                current_text = node_inputs.get('text', '')
                # Determine prompt type by content or title
                is_negative = any(word in title.lower() for word in ['negative', 'neg']) or \
                             any(word in current_text.lower() for word in ['bad', 'worst', 'low quality'])
                
                inputs[f"text_{node_id}"] = {
                    'type': 'text',
                    'node_id': node_id,
                    'title': title,
                    'value': current_text,
                    'multiline': True,
                    'prompt_type': 'negative' if is_negative else 'positive',
                    'required': True
                }
                
            # KSampler settings
            elif class_type == 'KSampler':
                # Seed
                inputs[f"seed_{node_id}"] = {
                    'type': 'number',
                    'node_id': node_id,
                    'title': f"{title} - Seed",
                    'value': node_inputs.get('seed', -1),
                    'min': -1,
                    'max': 2147483647,
                    'required': False
                }
                
                # Steps
                inputs[f"steps_{node_id}"] = {
                    'type': 'number',
                    'node_id': node_id,
                    'title': f"{title} - Steps",
                    'value': node_inputs.get('steps', 20),
                    'min': 1,
                    'max': 150,
                    'required': False
                }
                
                # CFG
                inputs[f"cfg_{node_id}"] = {
                    'type': 'number',
                    'node_id': node_id,
                    'title': f"{title} - CFG Scale",
                    'value': node_inputs.get('cfg', 7),
                    'min': 1,
                    'max': 30,
                    'step': 0.5,
                    'required': False
                }
                
            # Empty Latent Image (dimensions)
            elif class_type == 'EmptyLatentImage':
                inputs[f"width_{node_id}"] = {
                    'type': 'number',
                    'node_id': node_id,
                    'title': f"{title} - Width",
                    'value': node_inputs.get('width', 512),
                    'min': 64,
                    'max': 2048,
                    'step': 64,
                    'required': False
                }
                
                inputs[f"height_{node_id}"] = {
                    'type': 'number',
                    'node_id': node_id,
                    'title': f"{title} - Height",
                    'value': node_inputs.get('height', 512),
                    'min': 64,
                    'max': 2048,
                    'step': 64,
                    'required': False
                }
                
        return inputs
        
    def apply_inputs(self, workflow: Dict[str, Any], user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply user inputs to workflow"""
        # Deep copy workflow
        import copy
        workflow_copy = copy.deepcopy(workflow)
        
        logger.info(f"Applying {len(user_inputs)} user inputs to workflow")
        
        for input_id, value in user_inputs.items():
            parts = input_id.split('_', 1)
            if len(parts) != 2:
                logger.warning(f"Invalid input ID format: {input_id}")
                continue
                
            input_type, node_id = parts
            
            if node_id not in workflow_copy:
                logger.warning(f"Node {node_id} not found in workflow")
                continue
                
            node = workflow_copy[node_id]
            logger.debug(f"Applying {input_type}={value} to node {node_id}")
            
            if input_type == 'image':
                node['inputs']['image'] = value
                logger.info(f"Set image for node {node_id}: {value}")
            elif input_type == 'text':
                node['inputs']['text'] = value
                logger.info(f"Set text for node {node_id}: {value[:50]}...")
            elif input_type == 'seed':
                seed_value = int(value)
                # If seed is -1, generate random seed
                if seed_value == -1:
                    import random
                    seed_value = random.randint(0, 2147483647)
                node['inputs']['seed'] = seed_value
                logger.info(f"Set seed for node {node_id}: {seed_value}")
            elif input_type == 'steps':
                node['inputs']['steps'] = int(value)
                logger.info(f"Set steps for node {node_id}: {value}")
            elif input_type == 'cfg':
                node['inputs']['cfg'] = float(value)
                logger.info(f"Set CFG for node {node_id}: {value}")
            elif input_type == 'width':
                node['inputs']['width'] = int(value)
                logger.info(f"Set width for node {node_id}: {value}")
            elif input_type == 'height':
                node['inputs']['height'] = int(value)
                logger.info(f"Set height for node {node_id}: {value}")
                
        # Log final workflow for debugging
        logger.debug(f"Final workflow has {len(workflow_copy)} nodes")
        
        # Validate workflow has required nodes
        has_output = False
        for node_id, node in workflow_copy.items():
            if node.get('class_type') in ['SaveImage', 'PreviewImage']:
                has_output = True
                break
                
        if not has_output:
            logger.warning("Workflow has no SaveImage or PreviewImage nodes!")
            
        return workflow_copy

# Initialize components
workflow_manager = WorkflowManager()
comfy_client = ComfyUIClient(config.comfyui_url)

# Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# HTML Template (keeping the same modern UI)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ComfyUI Studio</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
            padding: 30px;
            border-radius: 16px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        
        h1 {
            font-size: 3em;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #888;
            font-size: 1.1em;
        }
        
        .workflow-selector {
            background: #1a1a1a;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        }
        
        select {
            width: 100%;
            padding: 15px 20px;
            font-size: 16px;
            background: #252525;
            color: #fff;
            border: 2px solid #333;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        select:hover {
            border-color: #667eea;
        }
        
        select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .workflow-description {
            margin-top: 15px;
            padding: 15px;
            background: #252525;
            border-radius: 8px;
            font-size: 0.95em;
            color: #aaa;
            display: none;
        }
        
        .workflow-count {
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
        }
        
        .inputs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .input-card {
            background: #1a1a1a;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .input-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        }
        
        .input-card h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.2em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .input-card h3::before {
            content: '';
            width: 4px;
            height: 20px;
            background: #667eea;
            border-radius: 2px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #aaa;
            font-size: 0.9em;
            font-weight: 500;
        }
        
        input[type="text"],
        input[type="number"],
        textarea {
            width: 100%;
            padding: 12px 15px;
            background: #252525;
            color: #fff;
            border: 2px solid #333;
            border-radius: 8px;
            font-size: 15px;
            transition: all 0.3s ease;
        }
        
        input[type="text"]:focus,
        input[type="number"]:focus,
        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        textarea {
            resize: vertical;
            min-height: 120px;
            font-family: inherit;
        }
        
        .file-upload {
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px;
            background: #252525;
            border: 3px dashed #444;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .file-upload:hover {
            border-color: #667eea;
            background: #2a2a2a;
        }
        
        .file-upload.has-file {
            border-color: #764ba2;
            background: #2a1f3d;
        }
        
        .file-upload input[type="file"] {
            position: absolute;
            opacity: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }
        
        .file-upload-icon {
            font-size: 48px;
            margin-bottom: 10px;
            opacity: 0.6;
        }
        
        .file-upload-text {
            color: #aaa;
            text-align: center;
        }
        
        .image-preview {
            margin-top: 15px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .image-preview img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .generate-section {
            background: #1a1a1a;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
            margin-bottom: 30px;
        }
        
        .generate-button {
            width: 100%;
            padding: 18px 40px;
            font-size: 18px;
            font-weight: 600;
            color: white;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .generate-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        
        .generate-button:disabled {
            background: #444;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .generate-button .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
            margin-left: 10px;
            vertical-align: middle;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .progress-section {
            background: #1a1a1a;
            padding: 25px;
            border-radius: 12px;
            margin-top: 20px;
            display: none;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        }
        
        .progress-section.active {
            display: block;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 15px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .progress-text {
            color: #aaa;
            font-size: 0.95em;
        }
        
        .results-section {
            margin-top: 40px;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .result-card {
            background: #1a1a1a;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .result-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        }
        
        .result-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .result-actions {
            padding: 15px;
            display: flex;
            gap: 10px;
        }
        
        .result-actions button {
            flex: 1;
            padding: 10px;
            background: #252525;
            color: #fff;
            border: 1px solid #333;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 14px;
        }
        
        .result-actions button:hover {
            background: #333;
            border-color: #667eea;
        }
        
        .error-message {
            background: #2d1515;
            color: #ff6b6b;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #ff4444;
            margin-top: 20px;
        }
        
        .info-message {
            background: #152d2d;
            color: #4ecdc4;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #4ecdc4;
            margin-bottom: 20px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
            background: #4CAF50;
        }
        
        .status-indicator.error {
            background: #f44336;
        }
        
        .status-indicator.warning {
            background: #ff9800;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
            background: #4CAF50;
        }
        
        .status-indicator.error {
            background: #f44336;
        }
        
        .status-indicator.warning {
            background: #ff9800;
        }
            .inputs-grid {
                grid-template-columns: 1fr;
            }
            
            h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ComfyUI Studio</h1>
            <p class="subtitle">Professional AI Image Generation Interface</p>
        </header>
        
        <div class="workflow-selector">
            <label for="workflow-select">Select Workflow:</label>
            <select id="workflow-select" onchange="loadWorkflow()">
                <option value="">-- Choose a workflow --</option>
            </select>
            <div id="workflow-count" class="workflow-count"></div>
            <div id="workflow-description" class="workflow-description"></div>
        </div>
        
        <div id="inputs-container" class="inputs-grid"></div>
        
        <div id="generate-section" class="generate-section" style="display: none;">
            <button id="generate-btn" class="generate-button" onclick="generate()">
                <span id="btn-text">Generate Image</span>
                <span id="btn-spinner" class="spinner" style="display: none;"></span>
            </button>
            
            <div id="progress-section" class="progress-section">
                <div id="progress-text" class="progress-text">Initializing...</div>
                <div class="progress-bar">
                    <div id="progress-fill" class="progress-fill"></div>
                </div>
            </div>
        </div>
        
        <div id="results-section" class="results-section"></div>
    </div>

    <script>
        let currentWorkflow = null;
        let uploadedFiles = {};
        let currentJob = null;

        // Load workflows on page load
        window.addEventListener('DOMContentLoaded', loadWorkflows);

        async function loadWorkflows() {
            try {
                const response = await fetch('/api/workflows');
                const workflows = await response.json();
                
                const select = document.getElementById('workflow-select');
                const countDiv = document.getElementById('workflow-count');
                
                workflows.forEach(wf => {
                    const option = document.createElement('option');
                    option.value = wf.id;
                    option.textContent = wf.name;
                    if (wf.filename) {
                        option.title = wf.filename;
                    }
                    select.appendChild(option);
                });
                
                countDiv.innerHTML = `<span class="status-indicator"></span>${workflows.length} workflows available`;
                
                if (workflows.length === 0) {
                    countDiv.innerHTML = `<span class="status-indicator warning"></span>No workflows found - add JSON files to the workflows folder`;
                }
                
            } catch (error) {
                console.error('Error loading workflows:', error);
                showError('Failed to load workflows');
            }
        }

        async function loadWorkflow() {
            const workflowId = document.getElementById('workflow-select').value;
            const container = document.getElementById('inputs-container');
            const generateSection = document.getElementById('generate-section');
            const descriptionDiv = document.getElementById('workflow-description');
            
            if (!workflowId) {
                container.innerHTML = '';
                generateSection.style.display = 'none';
                descriptionDiv.style.display = 'none';
                currentWorkflow = null;
                uploadedFiles = {};
                return;
            }

            try {
                const response = await fetch(`/api/workflows/${workflowId}`);
                const data = await response.json();
                
                currentWorkflow = data;
                uploadedFiles = {};
                
                // Show description
                if (data.description) {
                    descriptionDiv.textContent = data.description;
                    if (data.filename) {
                        descriptionDiv.textContent += ` (${data.filename})`;
                    }
                    descriptionDiv.style.display = 'block';
                } else {
                    descriptionDiv.style.display = 'none';
                }
                
                // Build input UI
                container.innerHTML = '';
                const inputsByType = groupInputsByType(data.inputs);
                
                // Render inputs by type
                for (const [type, inputs] of Object.entries(inputsByType)) {
                    if (inputs.length === 0) continue;
                    
                    const card = document.createElement('div');
                    card.className = 'input-card';
                    
                    const title = document.createElement('h3');
                    title.textContent = getTypeTitle(type);
                    card.appendChild(title);
                    
                    inputs.forEach(input => {
                        const formGroup = createInputElement(input.id, input);
                        card.appendChild(formGroup);
                    });
                    
                    container.appendChild(card);
                }
                
                generateSection.style.display = 'block';
                
            } catch (error) {
                console.error('Error loading workflow:', error);
                showError('Failed to load workflow details');
            }
        }

        function groupInputsByType(inputs) {
            const groups = {
                image: [],
                positive_prompt: [],
                negative_prompt: [],
                generation_settings: [],
                dimensions: []
            };
            
            for (const [id, input] of Object.entries(inputs)) {
                if (input.type === 'hidden') continue;
                
                if (input.type === 'image') {
                    groups.image.push({ id, ...input });
                } else if (input.type === 'text' && input.prompt_type === 'positive') {
                    groups.positive_prompt.push({ id, ...input });
                } else if (input.type === 'text' && input.prompt_type === 'negative') {
                    groups.negative_prompt.push({ id, ...input });
                } else if (id.includes('width') || id.includes('height')) {
                    groups.dimensions.push({ id, ...input });
                } else {
                    groups.generation_settings.push({ id, ...input });
                }
            }
            
            return groups;
        }

        function getTypeTitle(type) {
            const titles = {
                image: '📷 Input Images',
                positive_prompt: '✨ Positive Prompts',
                negative_prompt: '🚫 Negative Prompts',
                generation_settings: '⚙️ Generation Settings',
                dimensions: '📐 Image Dimensions'
            };
            return titles[type] || 'Settings';
        }

        function createInputElement(id, input) {
            const formGroup = document.createElement('div');
            formGroup.className = 'form-group';
            
            if (input.type === 'image') {
                const fileUpload = document.createElement('div');
                fileUpload.className = 'file-upload';
                fileUpload.id = `upload-${id}`;
                
                const fileInput = document.createElement('input');
                fileInput.type = 'file';
                fileInput.id = `input-${id}`;
                fileInput.accept = 'image/*';
                fileInput.onchange = (e) => handleFileSelect(e, id);
                
                const icon = document.createElement('div');
                icon.className = 'file-upload-icon';
                icon.textContent = '📁';
                
                const text = document.createElement('div');
                text.className = 'file-upload-text';
                text.textContent = 'Click or drag image here';
                
                fileUpload.appendChild(fileInput);
                fileUpload.appendChild(icon);
                fileUpload.appendChild(text);
                
                const preview = document.createElement('div');
                preview.id = `preview-${id}`;
                preview.className = 'image-preview';
                preview.style.display = 'none';
                
                formGroup.appendChild(fileUpload);
                formGroup.appendChild(preview);
                
            } else if (input.type === 'text') {
                const label = document.createElement('label');
                label.textContent = input.title;
                label.htmlFor = `input-${id}`;
                
                const textarea = document.createElement('textarea');
                textarea.id = `input-${id}`;
                textarea.value = input.value || '';
                textarea.placeholder = input.prompt_type === 'positive' 
                    ? 'Describe what you want to see...' 
                    : 'Describe what to avoid...';
                
                formGroup.appendChild(label);
                formGroup.appendChild(textarea);
                
            } else if (input.type === 'number') {
                const label = document.createElement('label');
                label.textContent = input.title;
                label.htmlFor = `input-${id}`;
                
                const numberInput = document.createElement('input');
                numberInput.type = 'number';
                numberInput.id = `input-${id}`;
                numberInput.value = input.value || 0;
                numberInput.min = input.min || 0;
                numberInput.max = input.max || 100;
                numberInput.step = input.step || 1;
                
                formGroup.appendChild(label);
                formGroup.appendChild(numberInput);
            }
            
            return formGroup;
        }

        function handleFileSelect(event, inputId) {
            const file = event.target.files[0];
            if (!file) return;
            
            uploadedFiles[inputId] = file;
            
            const uploadDiv = document.getElementById(`upload-${inputId}`);
            uploadDiv.classList.add('has-file');
            uploadDiv.querySelector('.file-upload-text').textContent = file.name;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                const preview = document.getElementById(`preview-${inputId}`);
                preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        async function generate() {
            const generateBtn = document.getElementById('generate-btn');
            const btnText = document.getElementById('btn-text');
            const btnSpinner = document.getElementById('btn-spinner');
            const progressSection = document.getElementById('progress-section');
            const progressText = document.getElementById('progress-text');
            const progressFill = document.getElementById('progress-fill');
            
            // Clear previous results
            document.getElementById('results-section').innerHTML = '';
            
            // Disable button and show spinner
            generateBtn.disabled = true;
            btnText.textContent = 'Generating...';
            btnSpinner.style.display = 'inline-block';
            progressSection.classList.add('active');
            progressFill.style.width = '0%';
            
            try {
                // Collect inputs
                const formData = new FormData();
                formData.append('workflow_id', currentWorkflow.id);
                
                const inputs = {};
                
                // Collect all inputs
                for (const [inputId, inputDef] of Object.entries(currentWorkflow.inputs)) {
                    if (inputDef.type === 'image') {
                        if (uploadedFiles[inputId]) {
                            formData.append(`file_${inputId}`, uploadedFiles[inputId]);
                        } else if (inputDef.required) {
                            throw new Error(`Please upload an image for ${inputDef.title}`);
                        }
                    } else {
                        const element = document.getElementById(`input-${inputId}`);
                        if (element) {
                            inputs[inputId] = element.value;
                        }
                    }
                }
                
                formData.append('inputs', JSON.stringify(inputs));
                
                // Start generation
                progressText.textContent = 'Uploading images...';
                progressFill.style.width = '20%';
                
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (!result.success) {
                    throw new Error(result.error || 'Generation failed');
                }
                
                currentJob = result.job_id;
                
                // Poll for status
                progressText.textContent = 'Processing...';
                progressFill.style.width = '40%';
                
                await pollJobStatus(result.job_id);
                
            } catch (error) {
                console.error('Generation error:', error);
                showError(error.message);
            } finally {
                generateBtn.disabled = false;
                btnText.textContent = 'Generate Image';
                btnSpinner.style.display = 'none';
                progressSection.classList.remove('active');
            }
        }

        async function pollJobStatus(jobId) {
            const progressText = document.getElementById('progress-text');
            const progressFill = document.getElementById('progress-fill');
            
            let attempts = 0;
            const maxAttempts = 600; // 5 minutes with 500ms polls
            
            while (attempts < maxAttempts) {
                try {
                    const response = await fetch(`/api/jobs/${jobId}`);
                    const job = await response.json();
                    
                    if (job.status === 'completed') {
                        progressText.textContent = 'Complete!';
                        progressFill.style.width = '100%';
                        displayResults(job.outputs);
                        return;
                    } else if (job.status === 'failed') {
                        throw new Error(job.error || 'Generation failed');
                    }
                    
                    // Update progress
                    const progress = Math.min(40 + (job.progress * 50), 90);
                    progressFill.style.width = `${progress}%`;
                    
                    if (job.current_node) {
                        progressText.textContent = `Processing node ${job.current_node}... ${Math.round(job.progress * 100)}%`;
                    } else {
                        progressText.textContent = `Processing... ${Math.round(job.progress * 100)}%`;
                    }
                    
                } catch (error) {
                    console.error('Status poll error:', error);
                }
                
                await new Promise(resolve => setTimeout(resolve, 500));
                attempts++;
            }
            
            throw new Error('Generation timed out');
        }

        function displayResults(outputs) {
            const resultsSection = document.getElementById('results-section');
            
            if (!outputs || outputs.length === 0) {
                showError('No output images generated');
                return;
            }
            
            resultsSection.innerHTML = `
                <h2 style="color: #667eea; margin-bottom: 20px;">Generated Images (${outputs.length})</h2>
                <div class="results-grid" id="results-grid"></div>
            `;
            
            const grid = document.getElementById('results-grid');
            
            outputs.forEach((output, index) => {
                const card = document.createElement('div');
                card.className = 'result-card';
                
                const img = document.createElement('img');
                img.src = `/api/outputs/${output.filename}`;
                img.alt = `Generated image ${index + 1}`;
                
                const actions = document.createElement('div');
                actions.className = 'result-actions';
                
                const downloadBtn = document.createElement('button');
                downloadBtn.textContent = '⬇ Download';
                downloadBtn.onclick = () => downloadImage(output.filename);
                
                const viewBtn = document.createElement('button');
                viewBtn.textContent = '🔍 View Full';
                viewBtn.onclick = () => window.open(`/api/outputs/${output.filename}`, '_blank');
                
                actions.appendChild(downloadBtn);
                actions.appendChild(viewBtn);
                
                card.appendChild(img);
                card.appendChild(actions);
                grid.appendChild(card);
            });
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        function downloadImage(filename) {
            const a = document.createElement('a');
            a.href = `/api/outputs/${filename}`;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }

        function showError(message) {
            const resultsSection = document.getElementById('results-section');
            resultsSection.innerHTML = `
                <div class="error-message">
                    <strong>Error:</strong> ${message}
                </div>
            `;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/workflows')
def api_workflows():
    """List all available workflows"""
    return jsonify(workflow_manager.workflows)
    # workflows = []
    # for wf_id, wf_data in workflow_manager.workflows.items():
    #     workflows.append({
    #         'id': wf_id,
    #         'name': wf_data['name'],
    #         'description': wf_data.get('description', ''),
    #         'filename': wf_data.get('filename', '')
    #     })
    # return jsonify(workflows)

@app.route('/api/workflows/<workflow_id>')
def api_workflow_details(workflow_id):
    """Get workflow details and required inputs"""
    workflow_data = workflow_manager.workflows.get(workflow_id)
    if not workflow_data:
        return jsonify({'error': 'Workflow not found'}), 404
    
    inputs = workflow_manager.analyze_workflow(workflow_data['workflow'])
    
    return jsonify({
        'id': workflow_id,
        'name': workflow_data['name'],
        'description': workflow_data.get('description', ''),
        'filename': workflow_data.get('filename', ''),
        'inputs': inputs
    })

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Generate images using a workflow"""
    try:
        workflow_id = request.form.get('workflow_id')
        if not workflow_id:
            return jsonify({'success': False, 'error': 'No workflow specified'}), 400
        
        workflow_data = workflow_manager.workflows.get(workflow_id)
        if not workflow_data:
            return jsonify({'success': False, 'error': 'Workflow not found'}), 404
        
        # Parse inputs
        inputs = json.loads(request.form.get('inputs', '{}'))

        sha256_hash = hashlib.sha256()

        # image urls
        img_urls = request.form.items()
        for key, img_url in img_urls:
            if key.startswith('img_url_'):
                input_id = key[8:]  # Remove 'img_url_' prefix

                try:
                    print(f"Downloading {img_url}")
                    response = requests.get(img_url, stream=True)
                    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

                    sha256_hash.update(img_url)
                    fname = sha256_hash.hexdigest()
                    fpath = os.path.join(config.UPLOAD_FOLDER, fname)
                    with open(fname, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    # Upload to ComfyUI
                    comfy_filename = comfy_client.upload_image(fpath, fname)
                    inputs[input_id] = comfy_filename
                    
                    logger.info(f"Uploaded {img_url} as {comfy_filename}")
                except requests.exceptions.HTTPError as e:
                    print(f"HTTP Error: {e}")
                except requests.exceptions.RequestException as e:
                    print(f"Request Error: {e}")

        # Handle file uploads
        # for key in request.files:
        #     if key.startswith('file_'):
        #         input_id = key[5:]  # Remove 'file_' prefix
        #         file = request.files[key]
                
        #         # Save uploaded file
        #         filename = secure_filename(file.filename)
        #         timestamp = str(int(time.time() * 1000))
        #         unique_filename = f"{timestamp}_{filename}"
        #         filepath = os.path.join(config.UPLOAD_FOLDER, unique_filename)
        #         file.save(filepath)
                
        #         # Upload to ComfyUI
        #         comfy_filename = comfy_client.upload_image(filepath, unique_filename)
        #         inputs[input_id] = comfy_filename
                
        #         logger.info(f"Uploaded {filename} as {comfy_filename}")
        
        # Apply inputs to workflow
        workflow = workflow_manager.apply_inputs(workflow_data['workflow'], inputs)
        
        # Queue the prompt
        prompt_id = comfy_client.queue_prompt(workflow)
        
        return jsonify({
            'success': True,
            'job_id': prompt_id
        })
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/jobs/<job_id>')
def api_job_status(job_id):
    """Get job status and wait for completion"""
    try:
        # Try to get the job from tracking first
        job = comfy_client.ws_client.jobs.get(job_id)
        
        if not job:
            # Try to get from history
            history = comfy_client.get_history(job_id)
            if job_id in history:
                # Create a job object from history
                job = GenerationJob(id=job_id, workflow={})
                job.status = 'completed'
                job.outputs = []
                
                # Get outputs from history
                outputs = history[job_id].get('outputs', {})
                for node_id, node_output in outputs.items():
                    if 'images' in node_output:
                        for img in node_output['images']:
                            job.outputs.append({
                                'filename': f"{job_id}_{img['filename']}",
                                'node_id': node_id
                            })
            else:
                return jsonify({'error': 'Job not found'}), 404
        
        # If job is still running, try to update from history
        if job.status in ['queued', 'running']:
            job = comfy_client.wait_for_job(job_id, timeout=1)  # Quick check
            
        return jsonify({
            'id': job.id,
            'status': job.status,
            'progress': job.progress,
            'outputs': job.outputs,
            'error': job.error,
            'current_node': job.current_node,
            'created_at': job.created_at.isoformat() if job.created_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None
        })
        
    except TimeoutError:
        # Return current status if timeout
        job = comfy_client.ws_client.jobs.get(job_id)
        if job:
            return jsonify({
                'id': job.id,
                'status': job.status,
                'progress': job.progress,
                'outputs': job.outputs,
                'error': job.error,
                'current_node': job.current_node
            })
        return jsonify({'error': 'Job timeout'}), 500
        
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/outputs/<filename>')
def api_output_image(filename):
    """Serve output images"""
    filepath = os.path.join(config.OUTPUT_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/png')
    return "Image not found", 404

if __name__ == '__main__':
    # Print startup message
    print("\n" + "="*70)
    print("ComfyUI Browser Interface")
    print("="*70)
    print(f"✓ ComfyUI URL: {config.comfyui_url}")
    print(f"✓ Web Interface: http://localhost:{config.PORT}")
    print(f"✓ Upload Folder: {config.UPLOAD_FOLDER}")
    print(f"✓ Output Folder: {config.OUTPUT_FOLDER}")
    print(f"✓ Workflows Folder: {config.WORKFLOWS_FOLDER}")
    print(f"✓ Loaded Workflows: {len(workflow_manager.workflows)}")
    print("="*70)
    print("Place your workflow JSON files in the 'workflows' folder")
    print("="*70 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=config.PORT,
        debug=True
    )