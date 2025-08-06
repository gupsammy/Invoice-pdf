"""
Performance Monitoring Tool for PDF Processing Pipeline
Analyzes logs and provides real-time performance metrics
"""

import re
import time
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

class PerformanceMonitor:
    """Monitor and analyze performance metrics from log files"""
    
    def __init__(self, log_file_path: str = "logs/invoice_extraction_2step_enhanced.log"):
        self.log_file_path = log_file_path
        self.metrics = {
            "classification_times": [],
            "extraction_times": [],
            "retry_counts": {},
            "error_patterns": {},
            "files_per_second": [],
            "api_response_times": []
        }
    
    def parse_log_file(self) -> Dict:
        """Parse log file and extract performance metrics"""
        if not Path(self.log_file_path).exists():
            print(f"Log file not found: {self.log_file_path}")
            return self.metrics
        
        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()
        
        start_time = None
        file_times = {}
        
        for line in lines:
            # Extract timestamp
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                
                if start_time is None:
                    start_time = timestamp
                
                # Track classification attempts
                if '[CLASSIFY]' in line:
                    self._parse_classification_line(line, timestamp, file_times)
                
                # Track extraction attempts
                elif '[EXTRACT]' in line:
                    self._parse_extraction_line(line, timestamp, file_times)
                
                # Track errors and retries
                elif 'Error' in line or 'Retry' in line:
                    self._parse_error_line(line)
        
        # Calculate statistics
        return self._calculate_statistics(file_times)
    
    def _parse_classification_line(self, line: str, timestamp: datetime, file_times: Dict):
        """Parse classification log lines"""
        # Extract file name and attempt info
        file_match = re.search(r'\[CLASSIFY\] ([\w\-\.]+) - Attempt (\d+)/(\d+)', line)
        if file_match:
            file_name = file_match.group(1)
            attempt = int(file_match.group(2))
            
            if file_name not in file_times:
                file_times[file_name] = {'start': timestamp, 'type': 'classification'}
        
        # Check for success
        success_match = re.search(r'\[CLASSIFY\] ([\w\-\.]+) - Success: (\w+) \(confidence: ([\d\.]+)\)', line)
        if success_match:
            file_name = success_match.group(1)
            if file_name in file_times and 'end' not in file_times[file_name]:
                file_times[file_name]['end'] = timestamp
                duration = (timestamp - file_times[file_name]['start']).total_seconds()
                self.metrics['classification_times'].append(duration)
    
    def _parse_extraction_line(self, line: str, timestamp: datetime, file_times: Dict):
        """Parse extraction log lines"""
        # Extract file name and attempt info
        file_match = re.search(r'\[EXTRACT\] ([\w\-\.]+) - Attempt (\d+)/(\d+)', line)
        if file_match:
            file_name = file_match.group(1)
            attempt = int(file_match.group(2))
            
            if file_name not in file_times:
                file_times[file_name] = {'start': timestamp, 'type': 'extraction'}
        
        # Check for success
        success_match = re.search(r'\[EXTRACT\] ([\w\-\.]+) - Success', line)
        if success_match:
            file_name = success_match.group(1)
            if file_name in file_times and 'end' not in file_times[file_name]:
                file_times[file_name]['end'] = timestamp
                duration = (timestamp - file_times[file_name]['start']).total_seconds()
                self.metrics['extraction_times'].append(duration)
    
    def _parse_error_line(self, line: str):
        """Parse error and retry lines"""
        # Count retry patterns
        if 'Retrying in' in line:
            retry_match = re.search(r'Retrying in (\d+)s', line)
            if retry_match:
                delay = int(retry_match.group(1))
                if delay not in self.metrics['retry_counts']:
                    self.metrics['retry_counts'][delay] = 0
                self.metrics['retry_counts'][delay] += 1
        
        # Track error patterns
        error_patterns = ['502', '429', 'timeout', 'JSON decode failed', 'INVALID_ARGUMENT']
        for pattern in error_patterns:
            if pattern in line:
                if pattern not in self.metrics['error_patterns']:
                    self.metrics['error_patterns'][pattern] = 0
                self.metrics['error_patterns'][pattern] += 1
    
    def _calculate_statistics(self, file_times: Dict) -> Dict:
        """Calculate comprehensive statistics"""
        stats = {
            'total_files_processed': len(file_times),
            'classification': {
                'count': len(self.metrics['classification_times']),
                'avg_time': statistics.mean(self.metrics['classification_times']) if self.metrics['classification_times'] else 0,
                'median_time': statistics.median(self.metrics['classification_times']) if self.metrics['classification_times'] else 0,
                'min_time': min(self.metrics['classification_times']) if self.metrics['classification_times'] else 0,
                'max_time': max(self.metrics['classification_times']) if self.metrics['classification_times'] else 0,
            },
            'extraction': {
                'count': len(self.metrics['extraction_times']),
                'avg_time': statistics.mean(self.metrics['extraction_times']) if self.metrics['extraction_times'] else 0,
                'median_time': statistics.median(self.metrics['extraction_times']) if self.metrics['extraction_times'] else 0,
                'min_time': min(self.metrics['extraction_times']) if self.metrics['extraction_times'] else 0,
                'max_time': max(self.metrics['extraction_times']) if self.metrics['extraction_times'] else 0,
            },
            'errors': self.metrics['error_patterns'],
            'retries': self.metrics['retry_counts'],
            'overall_avg_time_per_file': 0
        }
        
        # Calculate overall average time per file
        all_times = self.metrics['classification_times'] + self.metrics['extraction_times']
        if all_times:
            stats['overall_avg_time_per_file'] = statistics.mean(all_times)
        
        return stats
    
    def print_performance_report(self):
        """Print a formatted performance report"""
        stats = self.parse_log_file()
        
        print("\n" + "="*60)
        print("PDF PROCESSING PERFORMANCE REPORT")
        print("="*60)
        
        print(f"\n📊 OVERALL STATISTICS")
        print(f"  Total Files Processed: {stats['total_files_processed']}")
        print(f"  Average Time per File: {stats['overall_avg_time_per_file']:.2f} seconds")
        
        print(f"\n🏷️  CLASSIFICATION PERFORMANCE")
        print(f"  Files Classified: {stats['classification']['count']}")
        print(f"  Average Time: {stats['classification']['avg_time']:.2f} seconds")
        print(f"  Median Time: {stats['classification']['median_time']:.2f} seconds")
        print(f"  Min/Max Time: {stats['classification']['min_time']:.2f}s / {stats['classification']['max_time']:.2f}s")
        
        print(f"\n📋 EXTRACTION PERFORMANCE")
        print(f"  Files Extracted: {stats['extraction']['count']}")
        print(f"  Average Time: {stats['extraction']['avg_time']:.2f} seconds")
        print(f"  Median Time: {stats['extraction']['median_time']:.2f} seconds")
        print(f"  Min/Max Time: {stats['extraction']['min_time']:.2f}s / {stats['extraction']['max_time']:.2f}s")
        
        if stats['errors']:
            print(f"\n⚠️  ERROR PATTERNS")
            for error, count in stats['errors'].items():
                print(f"  {error}: {count} occurrences")
        
        if stats['retries']:
            print(f"\n🔄 RETRY PATTERNS")
            for delay, count in stats['retries'].items():
                print(f"  {delay}s delay: {count} retries")
        
        # Performance recommendations
        print(f"\n💡 PERFORMANCE INSIGHTS")
        
        avg_time = stats['overall_avg_time_per_file']
        if avg_time > 0:
            files_per_hour = 3600 / avg_time
            print(f"  Current throughput: {files_per_hour:.0f} files/hour")
            
            # Estimate for large datasets
            for dataset_size in [1000, 5000, 10000, 50000]:
                estimated_hours = (dataset_size * avg_time) / 3600
                print(f"  Estimated time for {dataset_size:,} files: {estimated_hours:.1f} hours")
        
        # Bottleneck analysis
        print(f"\n🔍 BOTTLENECK ANALYSIS")
        if stats['classification']['avg_time'] > stats['extraction']['avg_time']:
            print(f"  ⚠️  Classification is the bottleneck ({stats['classification']['avg_time']:.2f}s avg)")
            print(f"  💡 Consider increasing MAX_CONCURRENT_CLASSIFY")
        else:
            print(f"  ⚠️  Extraction is the bottleneck ({stats['extraction']['avg_time']:.2f}s avg)")
            print(f"  💡 Consider increasing MAX_CONCURRENT_EXTRACT")
        
        # Error rate analysis
        total_operations = stats['classification']['count'] + stats['extraction']['count']
        total_errors = sum(stats['errors'].values())
        if total_operations > 0:
            error_rate = (total_errors / total_operations) * 100
            print(f"\n📈 RELIABILITY METRICS")
            print(f"  Error Rate: {error_rate:.2f}%")
            print(f"  Success Rate: {100 - error_rate:.2f}%")
            
            if error_rate > 10:
                print(f"  ⚠️  High error rate detected! Consider:")
                print(f"     - Reducing concurrency limits")
                print(f"     - Implementing exponential backoff")
                print(f"     - Checking API quotas")
        
        print("\n" + "="*60)
        
        # Save report to JSON
        report_path = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n📁 Full report saved to: {report_path}")

def compare_performance(before_log: str, after_log: str):
    """Compare performance between two log files (before/after optimization)"""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Analyze both logs
    before_monitor = PerformanceMonitor(before_log)
    after_monitor = PerformanceMonitor(after_log)
    
    before_stats = before_monitor.parse_log_file()
    after_stats = after_monitor.parse_log_file()
    
    # Calculate improvements
    metrics = [
        ('Overall Avg Time', 
         before_stats['overall_avg_time_per_file'], 
         after_stats['overall_avg_time_per_file']),
        ('Classification Avg Time', 
         before_stats['classification']['avg_time'], 
         after_stats['classification']['avg_time']),
        ('Extraction Avg Time', 
         before_stats['extraction']['avg_time'], 
         after_stats['extraction']['avg_time']),
    ]
    
    print(f"\n{'Metric':<25} {'Before':>10} {'After':>10} {'Improvement':>15}")
    print("-" * 60)
    
    for metric_name, before_val, after_val in metrics:
        if before_val > 0:
            improvement = ((before_val - after_val) / before_val) * 100
            improvement_str = f"{improvement:+.1f}%" if improvement != 0 else "No change"
        else:
            improvement_str = "N/A"
        
        print(f"{metric_name:<25} {before_val:>10.2f}s {after_val:>10.2f}s {improvement_str:>15}")
    
    # Throughput comparison
    if before_stats['overall_avg_time_per_file'] > 0 and after_stats['overall_avg_time_per_file'] > 0:
        before_throughput = 3600 / before_stats['overall_avg_time_per_file']
        after_throughput = 3600 / after_stats['overall_avg_time_per_file']
        throughput_improvement = ((after_throughput - before_throughput) / before_throughput) * 100
        
        print(f"\n📊 THROUGHPUT COMPARISON")
        print(f"  Before: {before_throughput:.0f} files/hour")
        print(f"  After:  {after_throughput:.0f} files/hour")
        print(f"  Improvement: {throughput_improvement:+.1f}%")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Monitor current performance
    monitor = PerformanceMonitor()
    monitor.print_performance_report()
    
    # Example: Compare before and after optimization
    # compare_performance("logs/before_optimization.log", "logs/after_optimization.log")