- **1.INPUT:**
    - Các gói tin trong luồng gom theo 5-tuple (src_IP, src_port, dst_IP, dst_port, protocol)
        - Ba đặc trưng cốt lõi cho mỗi packet:
            - Δt: khoảng thời gian giữa hai packet liên tiếp (ms)
            - length: kích thước packet (bytes)
            - direction: hướng truyền (0 = forward, 1 = backward)
    - Từ 3 đặc trưng thô này, có nhiều cách mã hóa thành spike inputs:
        - **Rate Coding:** Chuyển cường độ đầu vào thành tần suất phát spike (firing rate) hoặc số lượng spike trong một cửa sổ thời gian
            - Giá trị x ∈ [0,1] → tại mỗi timestep, phát spike với xác suất x (Poisson spike train), hoặc phát x*T spikes trong T timesteps.
            - -> Chuẩn hóa packet length và Δt về [0,1], sau đó tạo Poisson spike train. Direction thì đơn giản 0/1
        - **Latency / Time-to-First-Spike (TTFS) Coding:** Chuyển cường độ thành thời điểm phát spike duy nhất. Giá trị lớn → spike sớm, giá trị nhỏ → spike muộn
            - 
            $$
            \begin{aligned}
            t_{\text{spike}} &= T_{\max}\left(1-\frac{x}{x_{\max}}\right)
            && \text{(Linear encoding)} \\[10pt]
            t_{\text{spike}} &= -\tau\ln\left(\frac{x}{x_{\max}}\right)
            && \text{(Logarithmic encoding)}
            \end{aligned}
            $$
            - Packet length lớn → spike sớm hơn. Δt nhỏ (gói đến nhanh) → spike sớm hơn
        - **Delta Modulation:** Chỉ phát spike khi có SỰ THAY ĐỔI đáng kể giữa hai packe liên tiếp, giữ im lặng khi ổn định
            - 
            ```
            if |length[t] - length[t-1]| > threshold_delta:
                spike = +1 (tăng) hoặc -1 (giảm)
            else:
                spike
        - **Stochastic Temporal Encoding:** Kết hợp temporal coding + stochastic — neuron phát spike với xác suất phụ thuộc membrane potential, tại thời điểm được xác định bởi giá trị input.
            - 
            $$
            P(\text{spike at } t)
            = \sigma\left(V(t) - \theta\right)
            = \frac{1}{1 + e^{-\left(V(t)-\theta\right)}}
            $$
        - **Spike-Time-Dependent Integrated (STDI) Coding:** Nén thông tin temporal vào ít timesteps hơn bằng cách tích hợp cả thời điểm và cường độ spike.
            - Giảm số timesteps inference → giảm latency trên FPGA.

- **2.Mô hình:**
    - **FEEDFORWARD SNN VỚI SURROGATE GRADIENT:**
        - Kiến trúc tổng thể: Input → SNN Layer 1 → SNN Layer 2 → ... → Output Layer
            - Cấu trúc:
                - 1-3 hidden layers fully-connected hoặc sparse
                - Mỗi layer: Linear transform → LIF neuron
            - Neuron dynamics (LIF):  
                $$
                U[t] = \beta U[t-1] + W X[t] - \theta S_{\text{out}}[t-1]
                $$
                $$
                S_{\text{out}}[t] =
                \begin{cases}
                1, & \text{nếu } U[t] > \theta, \\
                0, & \text{ngược lại}.
                \end{cases}
                $$
                - β = decay rate (learnable hoặc fixed)
            - Số neuron:
                - Input: tùy encoding 
                - Hidden: 128-512 neurons
                - Output: số class
            - Training: Surrogate gradient backpropagation 
    - **SNN VỚI LEARNABLE AXONAL DELAYS (DLTC):**
        - Tổng thể: Input → [Delay + Weight] Layer 1 → LIF → ... → Output
            - Cải tiến so với Feedforward SNN:
                - Mỗi synapse có delay d_ij có thể học được
                - Arrival time: 
                $$
                t_{\text{arrival}} = t_{\text{spike},j} + d_{ij}
                $$
                - Ngưỡng θ_i cũng learnable
            - Neuron dynamics:
                $$
                U_i[t] = \beta U_i[t-1] + \sum_j W_{ij} S_j[t-d_{ij}]
                $$
                $$
                S_i[t] =
                \begin{cases}
                1, & \text{nếu } U_i[t] > \theta_i,\\
                0, & \text{ngược lại}.
                \end{cases}
                $$
            - Training: Tối ưu đồng thời {W, d, θ} bằng surrogate gradient
    - **SNN AUTOENCODER:**
        - Tổng thể: Input → Encoder (SNN layers, giảm dần) → Bottleneck → Decoder → Output. Phân loại dựa trên reconstruction error hoặc bottleneck features
            - Encoder: Các lớp SNN nén dần (ví dụ 128→64→32)
            - Bottleneck: Biểu diễn nén (latent spikes)
            - Decoder: Tái tạo input (32→64→128)
            - Hai cách dùng cho classification:
                - a) Anomaly detection: reconstruction error cao → anomaly
                - b) Feature extraction: lấy bottleneck features → linear classifier
    - **NAS-OPTIMIZED MULTI-LIQUID LSM:**
        - Tổng thể: Input → [Liquid 1] → [Liquid 2] →  [Liquid 3] → Merge → Readout
            - Nhiều reservoir nhỏ (liquids) thay vì 1 reservoir lớn
            - Các liquid có thể song song hoặc nối tiếp
            - NAS + Simulated Annealing tìm cấu hình tối ưu:
                - Số liquid, kích thước mỗi liquid
                - Tỷ lệ kết nối, tỷ lệ excitatory/inhibitory
                - Cách kết nối giữa các liquid
    - **SNN VỚI STDP (UNSUPERVISED):**
        - Tổng thể: Input → STDP Layer(s) → Classifier (SVM/linear)
            - STDP (Spike-Timing-Dependent Plasticity):
                - **Long-Term Potentiation (LTP) – Pre trước Post:**

                $$
                \Delta W = A_{+} \exp\left(-\frac{\Delta t}{\tau_{+}}\right), \quad \Delta t > 0
                $$

                - **Long-Term Depression (LTD) – Post trước Pre:**

                $$
                \Delta W = -A_{-} \exp\left(\frac{\Delta t}{\tau_{-}}\right), \quad \Delta t < 0
                $$

            - Học không giám sát — weights tự điều chỉnh dựa trên
            tương quan thời gian giữa pre- và post-synaptic spikes

            - Sau STDP: lấy spike patterns → classifier đơn giản

    - **ANN-to-SNN CONVERSION + TIMESTEP COMPRESSION:**
        - Quy trình:
            - Train ANN thông thường (CNN/MLP) trên traffic data
            - Convert sang SNN: thay ReLU bằng IF neuron
                $$
                \text{Activation} = \text{Firing Rate} \times T
                $$
            - Nén timesteps: dùng STDI coding giảm T từ ~100 xuống ~4-8

- **3.OUTPUT:**
    - **Rate Decoding (Spike Count):**
        $$
        \hat{y} = \underset{c}{\arg\max} \left( \sum_{t=1}^{T} S_c[t] \right)
        $$
        - Mỗi output neuron ứng với 1 class. Đếm tổng số spike trong cửa sổ
        thời gian, class có nhiều spike nhất thắng.
        - Biến thể: Dùng membrane potential cuối cùng thay vì đếm spike:
        $$
        \hat{y} = \underset{c}{\arg\max} \left( U_c[T] \right)
        $$
    - **Latency Decoding (First Spike):**
        $$
        \hat{y} = \underset{c}{\arg\min} \left( t_{\text{first spike},c} \right)
        $$
        - Class có output neuron fire sớm nhất thắng.
    - **Population Decoding:**
        - Nhiều output neuron cho mỗi class (ví dụ 10 neuron/class).
        - Vote bằng tổng spike hoặc trung bình firing rate.
        $$
        \hat{y} = \underset{c}{\arg\max} \left( \sum_{i \in \mathcal{N}_c} N_i^{\text{spike}} \right)
        $$
    - **Linear Readout (Ridge Regression):**
        $$
        \hat{y} = \underset{c}{\arg\max} \left( W_{\text{read}} \mathbf{z} + \mathbf{b} \right)_c
        $$
        - W_read và b được train offline (pseudo-inverse hoặc ridge regression).
        - z = trạng thái cuối của reservoir.
- **4.Objective Functions:**
    - Spike Rate Objectives:
        - a) Cross-Entropy Loss trên spike rate:
            $$
            \mathcal{L} = -\sum_{c=1}^{C} y_c \log(p_c)
            $$
            - Trong đó:
            $$
            p_c = \operatorname{softmax}\left(\frac{N_c^{\text{spike}}}{T}\right)
            $$
            - Hoặc
            $$
            p_c = \operatorname{softmax}\left(U_c[T]\right)
            $$

        - b) Mean Square Error trên spike rate:
            $$
            \mathcal{L} = \sum_{c=1}^{C} (y_c - r_c)^2
            $$
            $$
            r_c = \frac{N_c^{\text{spike}}}{T}
            $$

    - Spike Time Objectives:
        - a) Cross-Entropy Loss trên spike time:
            $$
            \mathcal{L} =
            -\sum_{c=1}^{C} y_c
            \log\left(
            \frac{\exp(-t_c^{\text{spike}})}
            {\sum_{k=1}^{C} \exp(-t_k^{\text{spike}})}
            \right)
            $$
            - Spike sớm hơn → xác suất cao hơn cho class đó.

        - b) MSE trên spike time:
            $$
            \mathcal{L} = \sum_{c=1}^{C}
            \left(
            t_c^{\text{target}} - t_c^{\text{spike}}
            \right)^2
            $$
            - Target: class đúng spike sớm, class sai spike muộn hoặc không spike.

        - c) Temporal penalty:
            $$
            \mathcal{L} = \sum_{c=1}^{C}
            \max\left(0, t_c^{\text{spike}} - t^{\text{deadline}}\right)
            $$
            - Phạt neuron đúng class nếu fire quá muộn.

    - Reconstruction Loss:
        $$
        \mathcal{L}_{\text{MSE}} = \left\| \mathbf{x} - \hat{\mathbf{x}} \right\|_2^2
        $$

        $$
        \mathcal{L}_{\text{BCE}} =
        -\sum_{i=1}^{N}
        \left[
        x_i \log(\hat{x}_i)
        +
        (1-x_i)\log(1-\hat{x}_i)
        \right]
        $$

        - x = input spike train, x̂ = reconstructed output

    - Ridge Regression Loss:
        - 
        $$
        \mathcal{L}(W) =
        \left\| \mathbf{y} - W\mathbf{z} \right\|_2^2 + \alpha \left\| W \right\|_F^2
        $$  

        - Giải dạng đóng:
        $$
        W^* = \left(Z^\top Z + \alpha I\right)^{-1} Z^\top Y
        $$
        - Không cần gradient descent