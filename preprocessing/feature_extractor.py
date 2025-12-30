"""
Feature Extractor for IoT Network Traffic.
Extracts behavioral and temporal features for botnet detection.
Focus on early-stage attack indicators.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from scipy import stats


class FeatureExtractor:
    """
    Extracts network traffic features for IoT botnet detection.
    Focuses on early-stage indicators like scanning, probing, and initial C&C communication.
    """

    def __init__(
        self,
        window_size: float = 2.0,
        slide_step: float = 0.5,
        feature_dim: int = 15
    ):
        """
        Initialize feature extractor.

        Args:
            window_size: Time window in seconds
            slide_step: Sliding window step in seconds
            feature_dim: Number of features to extract
        """
        self.window_size = window_size
        self.slide_step = slide_step
        self.feature_dim = feature_dim

        # Feature names for reference
        self.feature_names = [
            'packet_count',
            'flow_duration',
            'syn_ack_ratio',
            'unique_dst_ip_count',
            'inter_arrival_time_mean',
            'inter_arrival_time_var',
            'payload_size_mean',
            'payload_size_var',
            'small_payload_ratio',
            'tcp_flags_entropy',
            'port_diversity',
            'bytes_per_second',
            'packets_per_second',
            'direction_ratio',
            'protocol_diversity'
        ]

    def extract_from_packets(
        self,
        packets: List[Dict],
        label: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from a list of packets using sliding windows.

        Args:
            packets: List of packet dictionaries
            label: Traffic label (0=normal, 1=botnet)

        Returns:
            Tuple of (features array, labels array)
        """
        if len(packets) == 0:
            return np.array([]), np.array([])

        # Sort packets by timestamp
        packets = sorted(packets, key=lambda x: x.get('timestamp', 0))

        start_time = packets[0].get('timestamp', 0)
        end_time = packets[-1].get('timestamp', 0)

        features_list = []
        labels_list = []

        current_time = start_time
        while current_time + self.window_size <= end_time:
            # Get packets in current window
            window_packets = [
                p for p in packets
                if current_time <= p.get('timestamp', 0) < current_time + self.window_size
            ]

            if len(window_packets) > 0:
                features = self._extract_window_features(window_packets)
                features_list.append(features)
                labels_list.append(label)

            current_time += self.slide_step

        if len(features_list) == 0:
            return np.array([]), np.array([])

        return np.array(features_list), np.array(labels_list)

    def _extract_window_features(self, packets: List[Dict]) -> np.ndarray:
        """
        Extract features from a single time window of packets.

        Args:
            packets: List of packets in the window

        Returns:
            Feature vector
        """
        features = np.zeros(self.feature_dim)

        if len(packets) == 0:
            return features

        # Basic statistics
        timestamps = [p.get('timestamp', 0) for p in packets]
        payload_sizes = [p.get('payload_size', 0) for p in packets]
        dst_ips = [p.get('dst_ip', '') for p in packets]
        dst_ports = [p.get('dst_port', 0) for p in packets]
        tcp_flags = [p.get('tcp_flags', 0) for p in packets]
        directions = [p.get('direction', 0) for p in packets]
        protocols = [p.get('protocol', 0) for p in packets]

        # 1. Packet count
        features[0] = len(packets)

        # 2. Flow duration
        if len(timestamps) > 1:
            features[1] = max(timestamps) - min(timestamps)
        else:
            features[1] = 0

        # 3. SYN/ACK ratio (indicator of scanning)
        syn_count = sum(1 for f in tcp_flags if f & 0x02)  # SYN flag
        ack_count = sum(1 for f in tcp_flags if f & 0x10)  # ACK flag
        if ack_count > 0:
            features[2] = syn_count / ack_count
        else:
            features[2] = syn_count if syn_count > 0 else 0

        # 4. Unique destination IP count (high value indicates scanning)
        features[3] = len(set(dst_ips))

        # 5. Inter-arrival time mean
        if len(timestamps) > 1:
            iat = np.diff(sorted(timestamps))
            features[4] = np.mean(iat) if len(iat) > 0 else 0
        else:
            features[4] = 0

        # 6. Inter-arrival time variance
        if len(timestamps) > 1:
            iat = np.diff(sorted(timestamps))
            features[5] = np.var(iat) if len(iat) > 0 else 0
        else:
            features[5] = 0

        # 7. Payload size mean
        features[6] = np.mean(payload_sizes) if payload_sizes else 0

        # 8. Payload size variance
        features[7] = np.var(payload_sizes) if len(payload_sizes) > 1 else 0

        # 9. Small payload ratio (indicator of C&C communication)
        small_threshold = 100  # bytes
        small_count = sum(1 for s in payload_sizes if s < small_threshold)
        features[8] = small_count / len(packets) if packets else 0

        # 10. TCP flags entropy (indicator of abnormal behavior)
        if tcp_flags:
            flag_counts = np.bincount(np.array(tcp_flags) % 256, minlength=256)
            flag_probs = flag_counts / sum(flag_counts)
            flag_probs = flag_probs[flag_probs > 0]
            features[9] = stats.entropy(flag_probs) if len(flag_probs) > 0 else 0
        else:
            features[9] = 0

        # 11. Port diversity (high diversity indicates scanning)
        features[10] = len(set(dst_ports))

        # 12. Bytes per second
        duration = features[1]
        total_bytes = sum(payload_sizes)
        features[11] = total_bytes / duration if duration > 0 else total_bytes

        # 13. Packets per second
        features[12] = len(packets) / duration if duration > 0 else len(packets)

        # 14. Direction ratio (outgoing vs incoming)
        outgoing = sum(1 for d in directions if d == 1)
        features[13] = outgoing / len(packets) if packets else 0.5

        # 15. Protocol diversity
        features[14] = len(set(protocols))

        return features

    def extract_early_stage_features(
        self,
        packets: List[Dict],
        early_window: float = 5.0,
        label: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features focusing only on early-stage traffic.
        This is critical for detecting botnets before full activation.

        Args:
            packets: List of packets
            early_window: Time window for early-stage (in seconds)
            label: Traffic label

        Returns:
            Tuple of (features, labels)
        """
        if len(packets) == 0:
            return np.array([]), np.array([])

        # Sort and filter to early-stage only
        packets = sorted(packets, key=lambda x: x.get('timestamp', 0))
        start_time = packets[0].get('timestamp', 0)

        early_packets = [
            p for p in packets
            if p.get('timestamp', 0) - start_time <= early_window
        ]

        return self.extract_from_packets(early_packets, label)

    def get_scanning_indicators(self, features: np.ndarray) -> Dict[str, float]:
        """
        Analyze features for scanning behavior indicators.

        Args:
            features: Feature vector

        Returns:
            Dictionary of scanning indicators
        """
        return {
            'high_syn_ratio': float(features[2] > 3.0),
            'port_scan_likelihood': min(1.0, features[10] / 100),
            'ip_scan_likelihood': min(1.0, features[3] / 50),
            'low_payload_indicator': features[8]
        }

    def get_cnc_indicators(self, features: np.ndarray) -> Dict[str, float]:
        """
        Analyze features for C&C communication indicators.

        Args:
            features: Feature vector

        Returns:
            Dictionary of C&C indicators
        """
        return {
            'small_payload_ratio': features[8],
            'low_traffic_rate': 1.0 - min(1.0, features[12] / 100),
            'outgoing_dominance': features[13],
            'regular_intervals': 1.0 - min(1.0, features[5] * 10)  # Low variance = regular
        }

    def normalize(
        self,
        features: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize features using min-max scaling.

        Args:
            features: Feature array
            fit: Whether to fit normalization parameters

        Returns:
            Normalized features
        """
        if fit:
            self.min_vals = features.min(axis=0)
            self.max_vals = features.max(axis=0)
            self.range_vals = self.max_vals - self.min_vals
            self.range_vals[self.range_vals == 0] = 1

        return (features - self.min_vals) / self.range_vals

    def get_feature_importance(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance using variance ratio.

        Args:
            features: Feature array
            labels: Label array

        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance = {}

        for i, name in enumerate(self.feature_names):
            normal_features = features[labels == 0, i]
            botnet_features = features[labels == 1, i]

            if len(normal_features) > 0 and len(botnet_features) > 0:
                # Use absolute difference of means normalized by pooled std
                pooled_std = np.sqrt(
                    (np.var(normal_features) + np.var(botnet_features)) / 2
                )
                if pooled_std > 0:
                    importance[name] = abs(
                        np.mean(botnet_features) - np.mean(normal_features)
                    ) / pooled_std
                else:
                    importance[name] = 0
            else:
                importance[name] = 0

        # Normalize importance scores
        max_imp = max(importance.values()) if importance.values() else 1
        if max_imp > 0:
            importance = {k: v / max_imp for k, v in importance.items()}

        return importance
