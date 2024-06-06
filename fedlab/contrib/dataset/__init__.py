# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .basic_dataset import FedDataset, BaseDataset, Subset
from .fcube import FCUBE
from .covtype import Covtype
from .rcv1 import RCV1

from .pathological_mnist import PathologicalMNIST
from .rotated_mnist import RotatedMNIST
from .rotated_cifar10 import RotatedCIFAR10
from .partitioned_mnist import PartitionedMNIST
from .partitioned_cifar10 import PartitionedCIFAR10
from .synthetic_dataset import SyntheticDataset
