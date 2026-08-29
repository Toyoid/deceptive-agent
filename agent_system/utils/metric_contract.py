# Copyright 2026 Hanxiao Li, Beihang University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared contract for environment-side episode metric logging.

- ``*_rate`` keys are logged as scalar env behavioral metrics.
- ``episode_metric/<name>`` keys are treated as per-trajectory episode metrics
  and reduced by the unified trainer logger into
  ``episode/<name>/{mean,std,max,min}``.
- ``episode_metric/<name_rate>`` keys are per-trajectory binary indicators and
  are reduced directly into ``episode/<name_rate>`` during training and
  ``val/<name_rate>`` during validation. Keeping the indicator per trajectory
  allows exact joint rates without reconstructing them from aggregate means.
"""


EPISODE_METRIC_PREFIX = "episode_metric/"
