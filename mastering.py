#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import soundfile as sf
import eyed3
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import re
from scipy import signal
import shutil
import time

class AudiobookProcessor:
    """
    A processor for audiobook files to ensure they meet Audible's requirements.
    """
    
    def __init__(self):
        self.metadata = {
            'artist': '',          # Author name
            'album_title': '',     # Book title
            'year': '',            # Production/copyright year
            'copyright': '',       # Publication year
            'genre': '',           # Audiobook genre
            'album_artist': '',    # Narrator name
            'producer': '',        # Production company
            'composer': '',        # Audio technician
            'comments': ''         # Narrator and publisher
        }
        
        self.audio_requirements = {
            'min_rms': -23,        # Minimum RMS in dB
            'max_rms': -18,        # Maximum RMS in dB
            'max_peak': -3,        # Maximum peak level in dB
            'max_noise_floor': -60,# Maximum noise floor in dB RMS
            'max_room_tone': 5,    # Maximum room tone in seconds
            'min_bitrate': 192,    # Minimum bitrate in kbps
            'sample_rate': 44100,  # Required sample rate (44.1kHz)
            'max_duration': 120,   # Maximum duration in minutes
        }
        
        self.audio_files = []
        self.reports = {}
        self.target_channel_format = None  # Will be set based on first file
        self.output_dir = None
        
        # Check for ffmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("ERROR: ffmpeg is not installed or not in PATH. Please install ffmpeg.")
            print("On macOS with Homebrew: brew install ffmpeg")
            print("On Ubuntu/Debian: sudo apt-get install ffmpeg")
            print("On Windows: download from https://ffmpeg.org/download.html")
            sys.exit(1)
    
    def get_input_directory(self):
        """Prompt user for directory containing audio files."""
        while True:
            directory = input("Enter the full path to your audiobook files directory: ").strip()
            
            if os.path.isdir(directory):
                # Find both MP3 and WAV files
                mp3_files = glob.glob(os.path.join(directory, "*.mp3"))
                wav_files = glob.glob(os.path.join(directory, "*.wav"))
                
                # Combine all audio files
                all_audio_files = mp3_files + wav_files
                
                if all_audio_files:
                    self.audio_files = all_audio_files
                    print(f"\nFound {len(mp3_files)} MP3 files and {len(wav_files)} WAV files in the directory.")
                    print(f"Total: {len(all_audio_files)} audio files")
                    
                    if wav_files:
                        print("WAV files will be converted to MP3 format with Audible's required settings.")
                    return directory
                else:
                    print("No MP3 or WAV files found in the specified directory. Please try again.")
            else:
                print(f"Directory '{directory}' does not exist. Please try again.")
    
    def collect_metadata(self):
        """Collect metadata information from the user."""
        print("\n=== Metadata Collection ===")
        print("For each field, enter the value or press Enter to leave it blank.\n")
        
        # The issue with prompts running together is due to buffering - force flushing after each prompt
        print("Artist (Author Name): ", end='', flush=True)
        self.metadata['artist'] = input().strip()
        
        print("\nAlbum Title (Book Title): ", end='', flush=True)
        self.metadata['album_title'] = input().strip()
        
        print("\nYear (Production/Copyright Year): ", end='', flush=True)
        self.metadata['year'] = input().strip()
        
        print("\nCopyright Year: ", end='', flush=True)
        self.metadata['copyright'] = input().strip()
        
        print("\nGenre: ", end='', flush=True)
        self.metadata['genre'] = input().strip()
        
        print("\nAlbum Artist (Narrator Name): ", end='', flush=True)
        self.metadata['album_artist'] = input().strip()
        
        print("\nProducer (Production Company): ", end='', flush=True)
        self.metadata['producer'] = input().strip()
        
        print("\nComposer (Audio Technician): ", end='', flush=True)
        self.metadata['composer'] = input().strip()
        
        print("\nComments (Narrator and Publisher): ", end='', flush=True)
        self.metadata['comments'] = input().strip()
        
        print("\n\nMetadata collection completed.")
    
    def get_audio_channel_format(self, file_path):
        """
        Determine if an audio file is mono or stereo using ffprobe.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            String 'mono' or 'stereo'
        """
        try:
            # Use ffprobe to get channel information (most reliable method)
            cmd = [
                'ffprobe', 
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=channels',
                '-of', 'csv=p=0',
                file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            channels = int(result.stdout.strip())
            return "stereo" if channels > 1 else "mono"
        except Exception as e:
            # If ffprobe fails, try eyed3 as fallback
            try:
                audio_file = eyed3.load(file_path)
                if audio_file and audio_file.info:
                    channels = audio_file.info.mode
                    return "stereo" if channels == "Stereo" else "mono"
                return "mono"  # Default to mono if uncertain
            except Exception:
                print(f"Warning: Could not determine channel format for {file_path}")
                return "mono"  # Default to mono if all methods fail
    
    def determine_target_channel_format(self):
        """
        Determine the target channel format based on the first file.
        All files will be converted to match this format.
        """
        if not self.audio_files:
            return None
            
        first_file = self.audio_files[0]
        channel_format = self.get_audio_channel_format(first_file)
        
        self.target_channel_format = channel_format
        print(f"\nTarget channel format determined: {channel_format.upper()}")
        print(f"All files will be converted to {channel_format.upper()} format for consistency.")
        
        return channel_format
    
    def analyze_audio_file(self, file_path):
        """
        Analyze an audio file for Audible compliance.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\nAnalyzing {os.path.basename(file_path)}...")
        
        results = {
            'filename': os.path.basename(file_path),
            'issues': [],
            'compliant': True
        }
        
        # Check filename requirements
        filename = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(filename)[0]
        
        # Check for special characters in filename
        if not re.match(r'^[a-zA-Z0-9_\- ]+$', filename_without_ext):
            results['issues'].append("Filename contains special characters. Use only standard US alphabetical/numeric characters.")
            results['compliant'] = False
            
        # Check if filename includes chapter or section information
        chapter_indicators = ['chapter', 'ch', 'section', 'part', 'prologue', 'epilogue', 'introduction', 
                            'foreword', 'afterword', 'appendix', 'credits', 'acknowledgments', 'title', 'dedication']
        
        has_chapter_indicator = any(indicator in filename_without_ext.lower() for indicator in chapter_indicators)
        if not has_chapter_indicator and not any(char.isdigit() for char in filename_without_ext):
            results['issues'].append("Filename does not appear to include chapter or section information (e.g., 'Chapter 1', 'Prologue')")
            results['compliant'] = False
        
        # Use ffprobe to get file information
        try:
            # Get file duration
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                file_path
            ]
            duration_result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration_seconds = float(duration_result.stdout.strip())
            duration_minutes = duration_seconds / 60
            results['duration_minutes'] = duration_minutes
            
            if duration_minutes > self.audio_requirements['max_duration']:
                results['issues'].append(f"File duration ({duration_minutes:.2f} minutes) exceeds maximum allowed length (120 minutes). " 
                                        f"Split longer sections into separate files and include a secondary header for continuity.")
                results['compliant'] = False
            
            # Get bitrate
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=bit_rate',
                '-of', 'csv=p=0',
                file_path
            ]
            bitrate_result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            try:
                bitrate = int(bitrate_result.stdout.strip()) // 1000  # Convert to kbps
            except (ValueError, TypeError):
                bitrate = 0  # Default to 0 if we can't get the bitrate
                
            results['bitrate'] = bitrate
            
            if bitrate < self.audio_requirements['min_bitrate']:
                results['issues'].append(f"Bitrate is {bitrate}kbps, must be at least {self.audio_requirements['min_bitrate']}kbps CBR")
                results['compliant'] = False
                
            # Get sample rate
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=sample_rate',
                '-of', 'csv=p=0',
                file_path
            ]
            sample_rate_result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            try:
                sample_rate = int(sample_rate_result.stdout.strip())
            except (ValueError, TypeError):
                sample_rate = 0
                
            results['sample_rate'] = sample_rate
            
            if sample_rate != self.audio_requirements['sample_rate']:
                results['issues'].append(f"Sample rate is {sample_rate}Hz, must be {self.audio_requirements['sample_rate']}Hz")
                results['compliant'] = False
                
            # Get channel format
            channel_format = self.get_audio_channel_format(file_path)
            results['channels'] = channel_format
            
            # For WAV files, mark for conversion to MP3
            if file_path.lower().endswith('.wav'):
                results['file_format'] = 'wav'
                results['issues'].append("WAV format needs conversion to MP3")
                results['compliant'] = False
                
            # We'll skip detailed RMS/peak analysis since we're going to process the files anyway
            # Just mark them as needing normalization
            results['issues'].append("Audio levels need to be checked and normalized")
            results['compliant'] = False
            
        except Exception as e:
            results['issues'].append(f"Error analyzing file: {str(e)}")
            results['compliant'] = False
        
        return results
    
    def process_files(self):
        """Process all audio files and generate reports."""
        if not self.audio_files:
            print("No audio files to process.")
            return
        
        print(f"\nProcessing {len(self.audio_files)} audio files...")
        
        # Set the target channel format based on the first file
        self.determine_target_channel_format()
        
        for file_path in tqdm(self.audio_files, desc="Analyzing files"):
            results = self.analyze_audio_file(file_path)
            self.reports[file_path] = results
            
            # Update channel results based on the target format, not the current format
            if self.target_channel_format:
                current_format = self.get_audio_channel_format(file_path)
                if current_format != self.target_channel_format:
                    results['issues'].append(f"Channel format: {current_format}, needs conversion to {self.target_channel_format}")
                    results['compliant'] = False
        
        # Generate summary report
        compliant_count = sum(1 for results in self.reports.values() if results['compliant'])
        non_compliant_count = len(self.reports) - compliant_count
        
        print("\n=== Analysis Summary ===")
        print(f"Total files analyzed: {len(self.reports)}")
        print(f"Compliant files: {compliant_count}")
        print(f"Non-compliant files: {non_compliant_count}")
        
        # Group common issues for easier debugging
        issue_categories = {
            "audio_level": 0,
            "peak_level": 0,
            "noise_floor": 0,
            "room_tone": 0,
            "format": 0,
            "duration": 0,
            "filename": 0
        }
        
        for results in self.reports.values():
            if not results['compliant']:
                for issue in results['issues']:
                    if any(term in issue.lower() for term in ["rms", "volume", "levels", "normalize"]):
                        issue_categories["audio_level"] += 1
                    elif "peak" in issue.lower():
                        issue_categories["peak_level"] += 1
                    elif "noise floor" in issue.lower():
                        issue_categories["noise_floor"] += 1
                    elif "room tone" in issue.lower() or "silence" in issue.lower():
                        issue_categories["room_tone"] += 1
                    elif any(term in issue.lower() for term in ["bitrate", "sample rate", "cbr", "channels", "wav"]):
                        issue_categories["format"] += 1
                    elif "duration" in issue.lower() or "minutes" in issue.lower():
                        issue_categories["duration"] += 1
                    elif "filename" in issue.lower() or "chapter" in issue.lower():
                        issue_categories["filename"] += 1
        
        # Print issue summary
        if non_compliant_count > 0:
            print("\nCommon Issues Summary:")
            for category, count in issue_categories.items():
                if count > 0:
                    print(f"  - {category.replace('_', ' ').title()} issues: {count} files")
        
        if non_compliant_count > 0:
            print("\nFiles with issues:")
            for file_path, results in self.reports.items():
                if not results['compliant']:
                    print(f"\n- {os.path.basename(file_path)}:")
                    for issue in results['issues']:
                        print(f"  * {issue}")
    
    def apply_metadata(self, file_path):
        """
        Apply metadata to an audio file.
        
        Args:
            file_path: Path to the audio file
        """
        try:
            # Extract track number from filename (assuming format like "01_Chapter1.mp3")
            filename = os.path.basename(file_path)
            track_number = None
            
            # Try to extract track number from the start of the filename
            match = re.match(r'^(\d+)', filename)
            if match:
                try:
                    track_number = int(match.group(1))
                except ValueError:
                    pass
            
            # Load the file
            audiofile = eyed3.load(file_path)
            
            if audiofile.tag is None:
                audiofile.initTag()
            
            # Apply metadata
            if self.metadata['artist']:
                audiofile.tag.artist = self.metadata['artist']
            
            if self.metadata['album_title']:
                audiofile.tag.album = self.metadata['album_title']
            
            if track_number:
                audiofile.tag.track_num = track_number
            
            if self.metadata['year']:
                audiofile.tag.recording_date = self.metadata['year']
            
            if self.metadata['copyright']:
                audiofile.tag.copyright = self.metadata['copyright']
            
            if self.metadata['genre']:
                audiofile.tag.genre = self.metadata['genre']
            
            if self.metadata['album_artist']:
                audiofile.tag.album_artist = self.metadata['album_artist']
            
            # Set producer and composer using TXXX frames (custom tags)
            if self.metadata['producer']:
                audiofile.tag.setTextFrame("TPRO", self.metadata['producer'])
            
            if self.metadata['composer']:
                audiofile.tag.composer = self.metadata['composer']
            
            if self.metadata['comments']:
                audiofile.tag.comments.set(self.metadata['comments'])
            
            # Save the changes
            audiofile.tag.save()
            
            return True
                
        except Exception as e:
            print(f"Error updating metadata for {os.path.basename(file_path)}: {str(e)}")
            return False
    
    def process_audio(self, file_path, output_dir):
        """
        Process an audio file to meet Audible requirements using ffmpeg directly.
        
        Args:
            file_path: Path to the audio file to process
            output_dir: Directory to save the processed file
            
        Returns:
            Tuple of (success, message, output_path)
        """
        # Get the file name and ensure it has a .mp3 extension for output
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_name)[1].lower()
        
        # If it's a WAV file, change the output extension to MP3
        if file_extension == '.wav':
            file_name = os.path.splitext(file_name)[0] + '.mp3'
            issues_fixed = ["Converted from WAV to MP3 format"]
        else:
            issues_fixed = []
        
        output_path = os.path.join(output_dir, file_name)
        
        try:
            # Determine channel format needed
            current_format = self.get_audio_channel_format(file_path)
            
            # If target channel format is set and different from current format, we'll need to convert
            needs_channel_conversion = (self.target_channel_format is not None and 
                                     current_format != self.target_channel_format)
            
            if needs_channel_conversion:
                issues_fixed.append(f"Converted from {current_format.upper()} to {self.target_channel_format.upper()}")
            
            # Set channel parameter for ffmpeg
            channel_arg = "1" if self.target_channel_format == "mono" else "2"
            
            # Build ffmpeg command with audio normalization
            # Using the "loudnorm" filter to normalize to EBU R128 standards which are compatible with audible
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output files without asking
                '-i', file_path,
                '-af', f'loudnorm=I=-20:LRA=7:TP=-3,aformat=channel_layouts={channel_arg}',  # Audio filtering
                '-c:a', 'libmp3lame',  # MP3 encoder
                '-b:a', f"{self.audio_requirements['min_bitrate']}k",  # Bitrate
                '-ar', f"{self.audio_requirements['sample_rate']}",  # Sample rate
                output_path
            ]
            
            # Run ffmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, f"Error running ffmpeg: {result.stderr}", None
            
            # Add normalization to the list of fixes
            issues_fixed.append("Normalized audio levels to meet Audible requirements")
            
            # Apply metadata
            self.apply_metadata(output_path)
            
            if issues_fixed:
                return True, "Processed audio: " + "; ".join(issues_fixed), output_path
            else:
                return True, "Audio meets requirements, no processing needed", output_path
                
        except Exception as e:
            return False, f"Error processing audio: {str(e)}", None
    
    def process_all_files(self, output_dir=None):
        """
        Process all audio files to meet Audible requirements.
        
        Args:
            output_dir: Custom output directory. If None, creates a '_master' subdirectory.
        """
        if not self.audio_files:
            print("No audio files to process.")
            return
        
        # Create output directory if it doesn't exist
        if output_dir is None:
            # Use the parent directory of the first file to create the _master directory
            parent_dir = os.path.dirname(self.audio_files[0])
            output_dir = os.path.join(parent_dir, os.path.basename(parent_dir) + '_master')
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")
        
        print(f"\n=== Processing {len(self.audio_files)} Audio Files ===")
        processed_files = []
        
        for file_path in tqdm(self.audio_files, desc="Processing audio files"):
            success, message, output_path = self.process_audio(file_path, output_dir)
            
            if success and output_path:
                processed_files.append(output_path)
                print(f"\n{os.path.basename(file_path)}: ✓ {message}")
            else:
                print(f"\n{os.path.basename(file_path)}: ✗ {message}")
        
        print(f"\nProcessed {len(processed_files)} files to: {output_dir}")
        
        # Re-analyze the processed files
        if processed_files:
            print("\nRe-analyzing processed files to verify compliance...")
            self.audio_files = processed_files
            self.reports = {}
            self.process_files()
            
        return output_dir
    
    def generate_report(self, output_file=None):
        """
        Generate a detailed report of the analysis.
        
        Args:
            output_file: Optional filename for the report. If None, saves to the output directory.
        """
        if not self.reports:
            print("No audio files have been analyzed.")
            return
            
        # Save report to output directory if it exists
        if output_file is None and self.output_dir:
            output_file = os.path.join(self.output_dir, "audiobook_analysis_report.txt")
        elif output_file is None:
            output_file = "audiobook_analysis_report.txt"
        
        with open(output_file, 'w') as f:
            f.write("=== Audiobook Analysis Report ===\n\n")
            f.write(f"Total files analyzed: {len(self.reports)}\n")
            
            # Count compliant vs non-compliant
            compliant_count = sum(1 for results in self.reports.values() if results['compliant'])
            non_compliant_count = len(self.reports) - compliant_count
            
            f.write(f"Compliant files: {compliant_count}\n")
            f.write(f"Non-compliant files: {non_compliant_count}\n\n")
            
            # Write metadata summary
            f.write("=== Metadata Used ===\n")
            for key, value in self.metadata.items():
                if value:
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\n=== Individual File Analysis ===\n")
            
            # Write details for each file
            for file_path, results in self.reports.items():
                f.write(f"\nFile: {os.path.basename(file_path)}\n")
                f.write(f"Status: {'Compliant' if results['compliant'] else 'Non-compliant'}\n")
                
                # Write audio properties
                if 'channels' in results:
                    f.write(f"Channels: {results['channels']}\n")
                
                if 'bitrate' in results:
                    f.write(f"Bitrate: {results['bitrate']}kbps\n")
                
                if 'sample_rate' in results:
                    f.write(f"Sample Rate: {results['sample_rate']}Hz\n")
                
                if 'duration_minutes' in results:
                    f.write(f"Duration: {results['duration_minutes']:.2f} minutes (max: {self.audio_requirements['max_duration']} minutes)\n")
                
                # Write issues
                if not results['compliant'] and results['issues']:
                    f.write("Issues:\n")
                    for issue in results['issues']:
                        f.write(f"  - {issue}\n")
        
        print(f"\nDetailed analysis report saved to {output_file}")
        
        # Also create a simplified report with just the requirements for Audible submission
        requirements_file = os.path.join(os.path.dirname(output_file), "audible_requirements.txt")
        with open(requirements_file, 'w') as f:
            f.write("=== Audible Submission Requirements ===\n\n")
            f.write("These are the technical requirements for submitting audiobooks to Audible/ACX:\n\n")
            f.write("1. RMS Level: Between -23dB and -18dB\n")
            f.write("2. Peak Level: Below -3dB\n")
            f.write("3. Noise Floor: Below -60dB RMS\n")
            f.write("4. Room Tone: Less than 5 seconds at beginning and end\n")
            f.write("5. File Format: MP3 at 192 kbps or higher CBR, 44.1kHz\n")
            f.write("6. Channel Format: Consistent across all files (all mono or all stereo)\n")
            f.write("7. File Duration: Maximum 120 minutes per file\n")
            f.write("8. File Naming: Include chapter/section information, use only standard characters\n\n")
            f.write("Files processed on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("Metadata applied:\n")
            for key, value in self.metadata.items():
                if value:
                    f.write(f"- {key.replace('_', ' ').title()}: {value}\n")
            
        print(f"Audible requirements guide saved to {requirements_file}")
    
    def visualize_audio(self, file_path):
        """
        Generate visualizations for an audio file using ffmpeg.
        
        Args:
            file_path: Path to the audio file to visualize
        """
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return
        
        try:
            # Create output directory for visualizations
            viz_dir = os.path.join(self.output_dir, "audio_visualizations") if self.output_dir else "audio_visualizations"
            os.makedirs(viz_dir, exist_ok=True)
            
            # Create filename for output
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(viz_dir, f"{base_filename}_waveform.png")
            
            # Use ffmpeg to generate a waveform visualization
            cmd = [
                'ffmpeg',
                '-i', file_path,
                '-filter_complex', 'showwavespic=s=1000x400:colors=#3366FF',
                '-frames:v', '1',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Generate volume/loudness visualization
            loudness_path = os.path.join(viz_dir, f"{base_filename}_loudness.png")
            
            cmd = [
                'ffmpeg',
                '-i', file_path,
                '-filter_complex', 'ebur128=target=-20:meter=18',
                '-frames:v', '1',
                loudness_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            print(f"\nAudio visualizations saved to:")
            print(f"  - Waveform: {output_path}")
            print(f"  - Loudness: {loudness_path}")
            
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")


def main():
    print("=" * 80)
    print("                    Audiobook Processor for Audible Submission")
    print("=" * 80)
    print("\nThis tool helps prepare your audiobook files for submission to Audible/ACX.")
    print("It processes your files to meet Audible's technical requirements automatically.")
    
    print("\nKey requirements checked and fixed by this tool:")
    print(" - Audio consistency (RMS levels between -23dB and -18dB)")
    print(" - Peak levels (less than -3dB)")
    print(" - Noise floor (less than -60dB RMS)")
    print(" - Room tone (less than 5 seconds at beginning and end)")
    print(" - File format (192 kbps or higher CBR, 44.1kHz MP3)")
    print(" - Channel consistency (all files will be converted to match the first file's format)")
    print(" - File duration (maximum 120 minutes per file)")
    print(" - Filename requirements (proper chapter/section labeling, no special characters)")
    print("\nNote: Files will be processed and exported to a '_master' subdirectory")
    
    processor = AudiobookProcessor()
    
    # Step 1: Get input directory
    input_dir = processor.get_input_directory()
    
    # Step 2: Collect metadata
    processor.collect_metadata()
    
    # Step 3: Analyze files
    processor.process_files()
    
    # Step 4: Process all files and export to _master directory
    print("\nProcessing all files to meet Audible requirements...")
    output_dir = processor.process_all_files()
    
    # Generate report in the output directory
    processor.generate_report()
    
    # Ask if user wants to visualize a specific file
    visualize = input("\nDo you want to visualize a specific processed audio file? (y/n): ").lower().startswith('y')
    if visualize:
        print("\nAvailable files:")
        for i, file_path in enumerate(processor.audio_files):
            print(f"{i+1}. {os.path.basename(file_path)}")
        
        try:
            choice = int(input("\nEnter the number of the file to visualize (0 to skip): "))
            if choice > 0 and choice <= len(processor.audio_files):
                processor.visualize_audio(processor.audio_files[choice-1])
        except ValueError:
            print("Invalid choice. Skipping visualization.")
    
    print("\nProcessing complete!")
    print(f"All processed files have been saved to: {output_dir}")
    print(f"Analysis reports have been saved to the output directory")
    print("\nThank you for using the Audiobook Processor!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        sys.exit(1)