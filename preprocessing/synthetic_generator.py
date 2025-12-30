"""
Synthetic IoT Botnet Traffic Generator.
Generates realistic synthetic data for training and testing when real datasets are unavailable.
Models both normal IoT traffic and various botnet behaviors.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import random


class SyntheticDataGenerator:
    """
    Generates synthetic IoT network traffic data for training and testing.
    Includes normal traffic and multiple botnet family patterns.
    """

    # Known botnet families and their characteristics
    BOTNET_FAMILIES = {
        'mirai': {
            'scan_rate': 0.8,
            'syn_ratio': 5.0,
            'payload_size_mean': 50,
            'payload_size_std': 20,
            'target_ports': [23, 2323, 80, 8080],
            'cnc_interval': 30
        },
        'bashlite': {
            'scan_rate': 0.6,
            'syn_ratio': 4.0,
            'payload_size_mean': 80,
            'payload_size_std': 30,
            'target_ports': [23, 22, 80],
            'cnc_interval': 60
        },
        'hajime': {
            'scan_rate': 0.4,
            'syn_ratio': 3.0,
            'payload_size_mean': 100,
            'payload_size_std': 40,
            'target_ports': [23, 5358, 7547],
            'cnc_interval': 120
        },
        'iot_reaper': {
            'scan_rate': 0.5,
            'syn_ratio': 3.5,
            'payload_size_mean': 120,
            'payload_size_std': 50,
            'target_ports': [80, 8080, 37215],
            'cnc_interval': 90
        },
        'zero_day': {  # Simulated unknown botnet
            'scan_rate': 0.7,
            'syn_ratio': 4.5,
            'payload_size_mean': 60,
            'payload_size_std': 25,
            'target_ports': [8888, 9999, 12345],
            'cnc_interval': 45
        }
    }

    def __init__(self, seed: int = 42):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_normal_traffic(
        self,
        num_flows: int = 1000,
        duration: float = 10.0
    ) -> List[Dict]:
        """
        Generate normal IoT device traffic.

        Args:
            num_flows: Number of traffic flows to generate
            duration: Duration of each flow in seconds

        Returns:
            List of packet dictionaries
        """
        packets = []

        for flow_id in range(num_flows):
            flow_packets = self._generate_normal_flow(flow_id, duration)
            packets.extend(flow_packets)

        return packets

    def _generate_normal_flow(self, flow_id: int, duration: float) -> List[Dict]:
        """Generate a single normal traffic flow."""
        packets = []
        base_time = flow_id * duration

        # Normal IoT traffic characteristics
        num_packets = np.random.poisson(20)  # Low packet rate
        src_ip = f"192.168.1.{random.randint(1, 254)}"
        dst_ip = f"10.0.0.{random.randint(1, 10)}"  # Limited destinations
        dst_port = random.choice([80, 443, 8883, 1883])  # Common IoT ports

        for i in range(num_packets):
            # Regular inter-arrival times with some variance
            timestamp = base_time + np.random.exponential(duration / num_packets) * i

            packet = {
                'flow_id': flow_id,
                'timestamp': timestamp,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': random.randint(49152, 65535),
                'dst_port': dst_port,
                'protocol': random.choice([6, 17]),  # TCP or UDP
                'payload_size': max(0, int(np.random.normal(200, 100))),
                'tcp_flags': random.choice([0x18, 0x10, 0x11]),  # PSH-ACK, ACK, FIN-ACK
                'direction': random.choice([0, 1]),
                'label': 0  # Normal
            }
            packets.append(packet)

        return packets

    def generate_botnet_traffic(
        self,
        botnet_family: str = 'mirai',
        num_flows: int = 1000,
        duration: float = 10.0,
        stage: str = 'early'
    ) -> List[Dict]:
        """
        Generate botnet traffic for a specific family.

        Args:
            botnet_family: Name of botnet family
            num_flows: Number of traffic flows
            duration: Duration of each flow
            stage: Attack stage ('early', 'scanning', 'cnc', 'attack')

        Returns:
            List of packet dictionaries
        """
        if botnet_family not in self.BOTNET_FAMILIES:
            raise ValueError(f"Unknown botnet family: {botnet_family}")

        config = self.BOTNET_FAMILIES[botnet_family]
        packets = []

        for flow_id in range(num_flows):
            if stage == 'early':
                flow_packets = self._generate_early_stage_flow(flow_id, duration, config)
            elif stage == 'scanning':
                flow_packets = self._generate_scanning_flow(flow_id, duration, config)
            elif stage == 'cnc':
                flow_packets = self._generate_cnc_flow(flow_id, duration, config)
            else:
                flow_packets = self._generate_attack_flow(flow_id, duration, config)

            packets.extend(flow_packets)

        return packets

    def _generate_early_stage_flow(
        self,
        flow_id: int,
        duration: float,
        config: Dict
    ) -> List[Dict]:
        """Generate early-stage botnet traffic (initial infection)."""
        packets = []
        base_time = flow_id * duration

        src_ip = f"192.168.1.{random.randint(1, 254)}"

        # Early stage: mix of scanning and initial C&C
        num_scan_packets = int(np.random.poisson(15) * config['scan_rate'])
        num_cnc_packets = np.random.poisson(5)

        # Scanning packets
        for i in range(num_scan_packets):
            timestamp = base_time + np.random.uniform(0, duration * 0.7)
            dst_ip = f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"

            packet = {
                'flow_id': flow_id,
                'timestamp': timestamp,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': random.randint(49152, 65535),
                'dst_port': random.choice(config['target_ports']),
                'protocol': 6,  # TCP
                'payload_size': max(0, int(np.random.normal(
                    config['payload_size_mean'],
                    config['payload_size_std']
                ))),
                'tcp_flags': 0x02,  # SYN
                'direction': 1,  # Outgoing
                'label': 1  # Botnet
            }
            packets.append(packet)

        # C&C communication packets
        cnc_ip = f"45.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        for i in range(num_cnc_packets):
            timestamp = base_time + np.random.uniform(duration * 0.5, duration)

            packet = {
                'flow_id': flow_id,
                'timestamp': timestamp,
                'src_ip': src_ip,
                'dst_ip': cnc_ip,
                'src_port': random.randint(49152, 65535),
                'dst_port': random.choice([80, 443, 8080]),
                'protocol': 6,
                'payload_size': random.randint(20, 100),  # Small C&C payloads
                'tcp_flags': 0x18,  # PSH-ACK
                'direction': 1,
                'label': 1
            }
            packets.append(packet)

        return packets

    def _generate_scanning_flow(
        self,
        flow_id: int,
        duration: float,
        config: Dict
    ) -> List[Dict]:
        """Generate scanning behavior traffic."""
        packets = []
        base_time = flow_id * duration

        src_ip = f"192.168.1.{random.randint(1, 254)}"
        num_packets = int(np.random.poisson(50) * config['scan_rate'])

        for i in range(num_packets):
            timestamp = base_time + i * (duration / num_packets)
            dst_ip = f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"

            packet = {
                'flow_id': flow_id,
                'timestamp': timestamp,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': random.randint(49152, 65535),
                'dst_port': random.choice(config['target_ports']),
                'protocol': 6,
                'payload_size': random.randint(0, 60),
                'tcp_flags': 0x02,  # SYN
                'direction': 1,
                'label': 1
            }
            packets.append(packet)

        return packets

    def _generate_cnc_flow(
        self,
        flow_id: int,
        duration: float,
        config: Dict
    ) -> List[Dict]:
        """Generate C&C communication traffic."""
        packets = []
        base_time = flow_id * duration

        src_ip = f"192.168.1.{random.randint(1, 254)}"
        cnc_ip = f"45.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

        # Regular beacon intervals
        interval = config['cnc_interval'] / 10  # Scaled for simulation
        num_beacons = int(duration / interval)

        for i in range(num_beacons):
            timestamp = base_time + i * interval + np.random.normal(0, 0.1)

            # Outgoing beacon
            packets.append({
                'flow_id': flow_id,
                'timestamp': timestamp,
                'src_ip': src_ip,
                'dst_ip': cnc_ip,
                'src_port': random.randint(49152, 65535),
                'dst_port': 443,
                'protocol': 6,
                'payload_size': random.randint(30, 80),
                'tcp_flags': 0x18,
                'direction': 1,
                'label': 1
            })

            # Incoming response
            packets.append({
                'flow_id': flow_id,
                'timestamp': timestamp + 0.05,
                'src_ip': cnc_ip,
                'dst_ip': src_ip,
                'src_port': 443,
                'dst_port': random.randint(49152, 65535),
                'protocol': 6,
                'payload_size': random.randint(50, 150),
                'tcp_flags': 0x18,
                'direction': 0,
                'label': 1
            })

        return packets

    def _generate_attack_flow(
        self,
        flow_id: int,
        duration: float,
        config: Dict
    ) -> List[Dict]:
        """Generate attack traffic (DDoS, etc.)."""
        packets = []
        base_time = flow_id * duration

        src_ip = f"192.168.1.{random.randint(1, 254)}"
        target_ip = f"203.0.113.{random.randint(1, 254)}"  # Target

        # High packet rate for attack
        num_packets = np.random.poisson(200)

        for i in range(num_packets):
            timestamp = base_time + i * (duration / num_packets)

            packet = {
                'flow_id': flow_id,
                'timestamp': timestamp,
                'src_ip': src_ip,
                'dst_ip': target_ip,
                'src_port': random.randint(1, 65535),
                'dst_port': random.choice([80, 443, 53]),
                'protocol': random.choice([6, 17]),
                'payload_size': random.randint(0, 1400),
                'tcp_flags': random.choice([0x02, 0x04, 0x14]),  # SYN, RST, RST-ACK
                'direction': 1,
                'label': 1
            }
            packets.append(packet)

        return packets

    def generate_dataset(
        self,
        num_normal: int = 2000,
        num_botnet: int = 2000,
        botnet_families: List[str] = None,
        include_zero_day: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate a complete dataset with normal and botnet traffic.

        Args:
            num_normal: Number of normal flows
            num_botnet: Number of botnet flows
            botnet_families: List of botnet families to include
            include_zero_day: Whether to include zero-day (unknown) samples

        Returns:
            Tuple of (training packets, test packets with zero-day)
        """
        if botnet_families is None:
            botnet_families = ['mirai', 'bashlite', 'hajime']

        # Generate normal traffic
        normal_packets = self.generate_normal_traffic(num_normal)

        # Generate botnet traffic from known families
        botnet_packets = []
        flows_per_family = num_botnet // len(botnet_families)

        for family in botnet_families:
            family_packets = self.generate_botnet_traffic(
                family,
                flows_per_family,
                stage='early'
            )
            botnet_packets.extend(family_packets)

        # Training set (known families)
        train_packets = normal_packets + botnet_packets

        # Test set with zero-day samples
        test_normal = self.generate_normal_traffic(num_normal // 4)
        test_botnet = self.generate_botnet_traffic(
            botnet_families[0],
            num_botnet // 4,
            stage='early'
        )

        test_packets = test_normal + test_botnet

        if include_zero_day:
            zero_day_packets = self.generate_botnet_traffic(
                'zero_day',
                num_botnet // 4,
                stage='early'
            )
            test_packets.extend(zero_day_packets)

        return train_packets, test_packets

    def generate_few_shot_episodes(
        self,
        n_episodes: int = 100,
        n_way: int = 2,
        k_shot: int = 5,
        n_query: int = 15,
        botnet_families: List[str] = None
    ) -> List[Dict]:
        """
        Generate episodes for few-shot meta-learning.

        Args:
            n_episodes: Number of episodes
            n_way: Number of classes per episode
            k_shot: Support samples per class
            n_query: Query samples per class
            botnet_families: Botnet families to use

        Returns:
            List of episode dictionaries
        """
        if botnet_families is None:
            botnet_families = list(self.BOTNET_FAMILIES.keys())

        episodes = []

        for _ in range(n_episodes):
            # Select a random botnet family for this episode
            family = random.choice(botnet_families)

            # Generate support set
            support_normal = self.generate_normal_traffic(k_shot, duration=5.0)
            support_botnet = self.generate_botnet_traffic(
                family, k_shot, duration=5.0, stage='early'
            )

            # Generate query set
            query_normal = self.generate_normal_traffic(n_query, duration=5.0)
            query_botnet = self.generate_botnet_traffic(
                family, n_query, duration=5.0, stage='early'
            )

            episode = {
                'support_normal': support_normal,
                'support_botnet': support_botnet,
                'query_normal': query_normal,
                'query_botnet': query_botnet,
                'botnet_family': family
            }
            episodes.append(episode)

        return episodes
