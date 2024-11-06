import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json
import os

class BrainValidator:
    def __init__(self, known_pumps_dir='yes', known_non_pumps_dir='no'):
        self.known_pumps = []
        self.known_non_pumps = []
        self.load_known_cases(known_pumps_dir, known_non_pumps_dir)
        
    def load_known_cases(self, pumps_dir, non_pumps_dir):
        """Load known classified cases"""
        # Load known pumps
        for price_file, enhanced_file in self._find_pairs(pumps_dir):
            self.known_pumps.append({
                'price_data': self._load_json(price_file),
                'enhanced_data': self._load_json(enhanced_file),
                'is_pump': True
            })
            
        # Load known non-pumps
        for price_file, enhanced_file in self._find_pairs(non_pumps_dir):
            self.known_non_pumps.append({
                'price_data': self._load_json(price_file),
                'enhanced_data': self._load_json(enhanced_file),
                'is_pump': False
            })
    
    def validate_brain(self, brain_model):
        """Run validation tests on the brain model"""
        results = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'misclassified_cases': []
        }
        
        # Test known pumps
        for case in self.known_pumps:
            prediction = brain_model.predict(case)
            if prediction:
                results['true_positives'] += 1
            else:
                results['false_negatives'] += 1
                results['misclassified_cases'].append({
                    'type': 'missed_pump',
                    'data': self._extract_key_metrics(case)
                })
        
        # Test known non-pumps
        for case in self.known_non_pumps:
            prediction = brain_model.predict(case)
            if prediction:
                results['false_positives'] += 1
                results['misclassified_cases'].append({
                    'type': 'false_pump',
                    'data': self._extract_key_metrics(case)
                })
            else:
                results['true_negatives'] += 1
        
        return self._analyze_results(results)
    
    def _extract_key_metrics(self, case):
        """Extract key metrics for analysis of misclassified cases"""
        metrics = {
            'price_metrics': {},
            'volume_metrics': {},
            'transaction_metrics': {},
            'account_metrics': {}
        }
        
        # Extract price movement
        prices = [x['price_in_usd'] for x in case['price_data']]
        metrics['price_metrics'] = {
            'initial_price': prices[0],
            'peak_price': max(prices),
            'price_increase': ((max(prices) - prices[0]) / prices[0]) * 100,
            'time_to_peak': len(prices)
        }
        
        # Extract volume metrics
        enhanced = case['enhanced_data']
        volumes = [w['window_metrics']['sol_metrics']['total_sol_volume'] for w in enhanced]
        metrics['volume_metrics'] = {
            'peak_volume': max(volumes),
            'avg_volume': np.mean(volumes),
            'volume_volatility': np.std(volumes)
        }
        
        # Transaction patterns
        tx_metrics = [w['window_metrics']['transaction_metrics'] for w in enhanced]
        metrics['transaction_metrics'] = {
            'max_tx_density': max(m['tx_density'] for m in tx_metrics),
            'avg_tx_density': np.mean([m['tx_density'] for m in tx_metrics]),
            'unique_accounts': max(m['unique_accounts'] for m in tx_metrics)
        }
        
        return metrics
    
    def _analyze_results(self, results):
        """Analyze validation results"""
        total_cases = (results['true_positives'] + results['false_positives'] + 
                      results['true_negatives'] + results['false_negatives'])
                      
        accuracy = (results['true_positives'] + results['true_negatives']) / total_cases
        precision = results['true_positives'] / (results['true_positives'] + results['false_positives'])
        recall = results['true_positives'] / (results['true_positives'] + results['false_negatives'])
        
        analysis = {
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': 2 * (precision * recall) / (precision + recall)
            },
            'confusion_matrix': {
                'true_positives': results['true_positives'],
                'false_positives': results['false_positives'],
                'true_negatives': results['true_negatives'],
                'false_negatives': results['false_negatives']
            },
            'misclassified_analysis': self._analyze_misclassified(results['misclassified_cases'])
        }
        
        return analysis
    
    def _analyze_misclassified(self, cases):
        """Analyze patterns in misclassified cases"""
        patterns = {
            'missed_pumps': {
                'avg_price_increase': [],
                'avg_volume_spike': [],
                'avg_tx_density': []
            },
            'false_pumps': {
                'avg_price_increase': [],
                'avg_volume_spike': [],
                'avg_tx_density': []
            }
        }
        
        for case in cases:
            category = 'missed_pumps' if case['type'] == 'missed_pump' else 'false_pumps'
            metrics = case['data']
            
            patterns[category]['avg_price_increase'].append(metrics['price_metrics']['price_increase'])
            patterns[category]['avg_volume_spike'].append(metrics['volume_metrics']['peak_volume'])
            patterns[category]['avg_tx_density'].append(metrics['transaction_metrics']['max_tx_density'])
        
        return patterns