# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Ngọc Hiệp  **MSSV**: 2A202601156  **Ngày**: 2026-08-21
**Tier**: `CPU`  **Base model**: `Qwen/Qwen3.5-0.8B`  **GPU thực tế**: RTX 3050 Ti Laptop 4 GB (Ampere `sm_86`)

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (corpus mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | **512** — p95 đo được là **98** token *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 → **58** optimizer step (effective batch 8) |
| Precision | `bf16` — chọn theo thiết bị, không hardcode |

**Template có giữ khối `<think>` không?** **Có** — `template_check.json` ghi *"reasoning preserved
— safe to train on traces"*. Nghĩa là nếu dữ liệu huấn luyện có chứa reasoning trace thật thì
chúng sẽ tới được hàm loss chứ không bị `apply_chat_template` nuốt mất. Trên corpus này điều đó
không phát huy tác dụng: cả 250 câu trả lời đều là JSON trần, và template Qwen3.5 đóng sẵn một
khối `<think></think>` rỗng **bên trong generation prompt**, nên `masked-think` và `response-only`
cho ra mask y hệt `assistant-only`. Lab cảnh báo đúng điều này khi chọn hai mode đó.

**Giải thích chênh lệch `max_length` (mục rubric 1.3).** p95 = 98 token, `max` = 101, và
`_round_pow2(98)` gợi ý 256; tier `CPU` đặt 512. Tôi **giữ 512** và đây là lý do tại sao lựa chọn
này không tốn gì: độ dài lớn nhất trong corpus là 101 token, nên **không mẫu nào bị cắt ở cả hai
mức**; và `per_device_train_batch_size = 1` nên mỗi chuỗi đứng một mình trong batch, **không có
padding nào để trả tiền**. `max_length` ở cấu hình này chỉ đóng vai trò trần cắt, và trần đó không
bao giờ chạm tới. Nếu tăng batch lên > 1, hoặc đổi sang corpus có đuôi dài, thì 256 mới là con số
đúng và 512 sẽ bắt đầu tốn thật.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.3936** (37 / 94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss:

```
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn **không** được tính loss (phần bù):

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Hai điều đáng chú ý ở đây, và cả hai đều không nhìn thấy được nếu chỉ tin vào một cờ thư viện:

1. **`<|im_end|>` nằm TRONG loss.** Đó là chủ ý — nó là tín hiệu dừng, và một model không được
   dạy dừng sẽ sinh tiếp cho tới khi cạn `max_new_tokens`. Chính xác là điều baseline (a) làm.
2. **Khối `<think></think>` rỗng nằm NGOÀI loss**, vì nó thuộc generation prompt chứ không thuộc
   nội dung câu trả lời. Đây là lý do kỹ thuật khiến ba mode mask không-`everything` trùng nhau
   trên corpus này.

Đối chứng bắt buộc: cùng mẫu đó, `MASK_MODE=everything` cho **94/94 token (100%)** — tức câu hỏi
nằm trong loss và model sẽ học cách viết lại câu hỏi (deck §16). Rubric trừ trắng mục 1.1 nếu
`supervised_fraction ≥ 0.95`; con số 0.3936 nằm an toàn.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

Đo trên **50 mẫu target / 15 mẫu regression**, `EVAL_LIMIT` không đặt.

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.667 | 0.000 | 2058.5 |
| (b) base + optimized prompt | **0.495** | 0.667 | **1.000** | **584.9** |
| **(c) LoRA fine-tune** | **0.985** | **0.033** | 1.000 | 752.7 |

**(b) có thật sự mạnh hơn (a) không?** Có, và cách biệt rất lớn: 0.495 so với 0.000. Tôi **không
sửa** `OPTIMIZED_PROMPT` — `verify.py` xác nhận SHA `719e74d3b6232053` khớp bản gốc.

Điều đáng nói hơn con số target: **prompt tốt vừa chính xác hơn vừa RẺ hơn**. (b) nhanh gấp 3,5
lần (a) — 585 ms so với 2058 ms — cùng một model, cùng phần cứng. Lý do nằm ở cột `format`: với
prompt ngây thơ, model 0.8B không đoán ra được schema 4 khoá nên nó viết văn xuôi cho tới khi cạn
160 token; với prompt có schema, nó phát ra JSON rồi dừng. Nghĩa là **(a) không phải một baseline
yếu một cách trung lập — nó yếu vì nó không được cho biết phải làm gì**, và toàn bộ 0.495 điểm kia
mua được bằng prompt engineering, chưa tốn một step huấn luyện nào.

Đây chính là mốc mà fine-tune phải vượt, và là lý do NB2 chạy trước NB3: nếu đo sau khi train, rất
khó cưỡng lại việc chỉnh (b) yếu đi cho tới khi bản fine-tune trông thắng.

---

## 4. Giải phẫu cấu hình sai (NB4)

Bốn run, **cùng 58 optimizer step**, **cùng ngân sách 10 822 656 tham số huấn luyện** (lệch
0,000%), mỗi run đổi đúng **một** biến.

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 10 822 656 | 1e-4 | **0.3956** | **0.985** | 1145.6 | 3.07 |
| `attn_only` | q,v | **271** *(matched)* | 10 822 656 | 1e-4 | 0.4333 | 0.935 | 1474.1 | 3.08 |
| `wrong_lr` | text-linear | 16 | 10 822 656 | **1e-5** | 1.5418 | **0.330** | 1876.1 | 3.08 |
| `qlora` | text-linear | 16 | 10 822 656 | 1e-4 | 0.4247 | 0.930 | 603.6 | **2.29** |

Biến được đổi ở từng run: `attn_only` = **vị trí gắn adapter**; `wrong_lr` = **learning rate**;
`qlora` = **độ chính xác lưu trọng số base (4-bit thay vì 16-bit)**. Mọi thứ còn lại giữ nguyên.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua,
hay hoà? Thứ tự đó có giống thứ tự theo train loss không?**

**`attn_only` THUA**, 0.935 so với 0.985 của `correct` — cách biệt 0.050 ở cùng ngân sách tham số
và cùng 58 step. Ở đây thứ tự theo `final_loss` (0.3956 < 0.4333) **trùng** với thứ tự theo target,
nên trong run này train loss tình cờ xếp hạng đúng. Nhưng tôi cũng chạy một lượt rút gọn
`EVAL_LIMIT=8` trước đó, và ở đó `attn_only` đạt **0.781** còn `correct` chỉ **0.719** — tức lát
eval nhỏ đảo ngược kết luận. Cùng một cặp adapter, cùng một thang đo, chỉ khác số mẫu. Bài học tôi
rút ra không phải "train loss luôn dối" mà là: **cả hai cột đều có thể dối, và ở lát 8 mẫu thì
chính cột target — cột được cho là đáng tin — mới là cột dối.**

Phần lý giải thì thuộc về kiến trúc chứ không phải kết quả:
base này xen kẽ **linear attention với full attention**, và `q_proj`/`v_proj` **chỉ tồn tại trên
các lớp full-attention**. Placement "chỉ q,v" vì thế chạm tới **12 module trên 6 trong 24 lớp**.
Để đưa nó về đúng ngân sách tham số của `correct`, `matched_rank()` phải nâng rank từ 16 lên
**271** — gấp 17 lần — mà **không mua thêm được một lớp nào**. Toàn bộ ngân sách bị dồn vào cùng
6 lớp đó, dày hơn thay vì rộng hơn. Đây là §10.2 ở dạng sắc hơn cả những gì deck phát biểu: trên
kiến trúc lai, rank không chỉ *kém hiệu quả hơn* vị trí, nó gần như **không thể thay thế** vị trí,
vì 18 lớp còn lại không có chỗ cho adapter bám vào.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao?**

Train loss dừng ở **1.5418**, gấp **3,9 lần** `correct` (0.3956), và target rơi xuống **0.330** —
mất 0.655 điểm chỉ vì LR ở thang full-fine-tune (1e-5) thay vì thang LoRA (1e-4). Đây là nút vặn
gây thiệt hại lớn nhất trong cả bốn run, gấp **13 lần** thiệt hại do gắn sai vị trí adapter.

Nếu chỉ nhìn loss mà không biết LR, tôi sẽ kết luận sai rằng **dữ liệu quá khó hoặc model quá nhỏ**
— vì đường loss của một run LR thấp trông y hệt đường loss của một bài toán khó: nó giảm, đều đặn,
chỉ là giảm chậm. Không có gì báo lỗi. Điều làm tôi tin chắc đó là chẩn đoán sai: cũng adapter đó ở
run rút gọn 29 step đạt **0.000 target và 0.000 format**, còn ở 58 step lên 0.330 / 0.995. LR nhỏ
hơn 10 lần không phá hỏng việc học, nó chỉ làm chậm khoảng 10 lần — nhưng ở một ngân sách step cố
định thì hệ quả là thua cả **baseline (b) = 0.495**, tức thua cả việc không huấn luyện gì cả.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì?**

VRAM đỉnh **2.29 GB** so với 3.07 GB của `correct` — **tiết kiệm 25,4%**. Trên card 4 GB của tôi
đó là khác biệt giữa "vừa khít" và "thoải mái". Nó còn train **nhanh hơn**: 603.6 s so với 1145.6 s,
vì base 4-bit làm giảm lưu lượng bộ nhớ, mà trên GPU laptop thì băng thông mới là nút cổ chai.

Giá phải trả có hai phần. Thứ nhất là **chất lượng**: target 0.930 so với 0.985 (−0.055) và format
0.980 so với 1.000 — nhỏ nhưng thật, và nhất quán với khuyến nghị của nhà cung cấp ở deck §12 rằng
sai số lượng tử hoá của dòng model này cao hơn bình thường. Thứ hai, và ít được nói tới hơn, là
**độ trễ khi phục vụ**: 1013.2 ms/mẫu so với 752.7 ms của `correct` — **chậm hơn 35%**, vì phải giải
lượng tử hoá trọng số ở mỗi lượt forward.

Nên số đo của tôi **ủng hộ khuyến nghị đó, nhưng có điều kiện**: QLoRA đổi VRAM lúc *huấn luyện*
lấy chất lượng *và* tốc độ lúc *phục vụ*. Nếu bạn không bị chặn bởi VRAM thì đó là một cuộc đổi chác
tệ. Với tôi ở 4 GB thì bf16 vẫn vừa, nên tôi chọn `correct`.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.490` · `regression Δ = −0.633` · `valid_trace_rate = 0.00`

Bản fine-tune **thắng áp đảo** ở đúng việc nó được huấn luyện: 0.985 so với 0.495 của prompt tốt
nhất, gần như hoàn hảo trên triage, và nhanh hơn baseline (a) 2,7 lần. Nếu lab chỉ chấm bằng target
thì đây là một thắng lợi lớn. Nó rớt ở cổng thứ hai: năng lực phổ thông sụp từ 0.667 xuống **0.033**,
tụt 0.633 so với ngưỡng cho phép 0.020 — vượt ngưỡng **31 lần**.

Nguyên nhân nằm ngay trong hình dạng dữ liệu, không phải trong siêu tham số. Model được huấn luyện
trên `ticket trần → JSON`, với một system prompt ngắn không mang thông tin gì phân biệt. Nó không
có cách nào học được rằng đây là *một* tác vụ trong nhiều tác vụ; thứ nó học được là **mọi input
đều có nghĩa là "phát ra JSON triage"**. Bằng chứng ở mục 6: hỏi nó viết một câu chúc sinh nhật,
nó trả về `{"intent": "hoi_thong_tin", ...}`.

Điều làm tôi tin đây là quên thảm hoạ thật chứ không phải model nhạy với hình dạng prompt: hai
nhóm được đo bằng **cùng một cách** cho cả base lẫn fine-tune, nên phép so sánh công bằng ở cả hai
phía; và mức sụp đổ **tỷ lệ thuận với mức độ khớp tác vụ** — ở 29 step target 0.719 / regression
0.250, ở 58 step target 0.985 / regression 0.033. Hai đường đi ngược chiều một cách có hệ thống,
đúng như deck §14.3 mô tả, và thuốc chữa mà §14.3 kê là 1–5% dữ liệu replay.

`valid_trace_rate = 0.00` không nói lên điều gì ở đây: cả 250 câu trả lời huấn luyện đều là JSON
trần, không có reasoning trace nào để mà sụp đổ. Chỉ số này chỉ có nghĩa cho bonus B3.

---

## 6. Định tính — bắt buộc có cả ca THUA

**Phải nói thẳng một điều trước khi lập bảng: trên tập target, fine-tune KHÔNG thua mẫu nào.**
Tôi so từng mẫu một giữa (b) và (c) trên cả 50 ticket (`results/head2head.json`): **48 thắng,
2 hoà, 0 thua**. Nên nếu tôi trình ra "hai ca fine-tune thua" lấy từ tập target thì đó là bịa.

Các ca thua có tồn tại, và chúng nằm ở **nhóm regression** — đúng chỗ mà cổng hồi quy chỉ ra.
Trên 15 câu hỏi phổ thông, fine-tune thua base ở **11/15** (`results/regression_head2head.json`).
Đó là bảng dưới đây, và nó là bức tranh trung thực về ca thua của model này.

**Ca fine-tune THUA (nhóm regression, base vs fine-tune):**

| # | Câu hỏi | base | (c) fine-tune | Nhận xét |
|---|---|---|---|---|
| 1 | *Viết một câu chúc mừng sinh nhật bằng tiếng Việt.* | **1.00** — "Chúc mừng sinh nhật! 🎉 Chúc bạn luôn tràn đầy năng lượng…" | **0.00** — `{"intent": "hoi_thong_tin", "urgency": "thap", "sentiment": "tieu_cuc"}` | ❌ **FT thua**. Câu hỏi sinh ngữ thuần tuý, không có gì để phân loại — model vẫn phát ra JSON. |
| 2 | *2 mũ 10 bằng bao nhiêu?* | **1.00** | **0.00** — `{"intent": "hoi_thong_tin", "urgency": "thap", "product": "hoi_thong_tin"}` | ❌ **FT thua**. Còn điền cả tên intent vào trường `product` — schema đã trở thành thứ duy nhất nó biết nói. |
| 3 | *1 km bằng bao nhiêu mét?* | **1.00** | **0.00** — `{"intent": "hoi_thong_tin", …, "product": "tich theo"}` | ❌ **FT thua**. `"tich theo"` không phải sản phẩm, cũng không phải câu trả lời — nó cố trích một "sản phẩm" ra khỏi một câu không có sản phẩm nào. |
| 4 | *Dịch sang tiếng Anh: 'Tôi thích đọc sách'.* | **1.00** — "I like reading books." | **0.50** — `{"intent": "reading", "urgency": "none", "sentiment": "positive"}` | ❌ **FT thua**. Ca thú vị nhất: nó *có* hiểu câu hỏi (đưa được "reading" ra), nhưng chỉ còn một khuôn duy nhất để trả lời, và cả enum cũng trôi sang tiếng Anh (`none`, `positive`). |
| 5 | *Kể tên hai đại dương lớn nhất thế giới.* | **0.50** | **0.00** — `{"intent": "hoi_thong_tin", "urgency": "thap", "product": "tieu name"}` | ❌ **FT thua**. |

**Ca fine-tune THẮNG (nhóm target, (b) vs (c)) — để đối chiếu:**

| # | Ticket (rút gọn) | Nhãn đúng (rút gọn) | (b) prompt | (c) fine-tune |
|---|---|---|---|---|
| 6 | *…nồi chiên không dầu… Thiếu phụ kiện. Khi nào tiện…* | `san_pham_loi` / `thap` / `trung_tinh` | **0.25** — đoán `hoan_tien` / `cao` / `tich_cuc` | **1.00** ✅ |
| 7 | *…balo laptop… Đổi size. Hỏi cho biết thôi…* | `doi_tra` / `thap` / `tieu_cuc` | **0.25** — đoán `hoan_tien` / `cao` / `tich_cuc` | **1.00** ✅ |

**Mẫu chung ở các ca (b) thua:** baseline (b) đoán `urgency` thành `cao` và `sentiment` thành
`tich_cuc` một cách gần như phản xạ, bất kể ticket nói gì. Nó đọc được `product` (chép nguyên văn,
dễ) nhưng không nắm được **thang mức độ** — thứ chỉ học được từ việc nhìn nhiều ví dụ đã gán nhãn,
chứ không mô tả nổi trong một prompt. Đó chính xác là thứ fine-tuning mua được ở đây.

**Mẫu chung ở các ca fine-tune thua:** không có ngoại lệ nào — **mọi** câu hỏi phổ thông đều bị trả
lời bằng JSON triage. Đây không phải nhiễu ở đuôi phân phối, mà là sự sụp đổ toàn diện và nhất quán.

---

## 7. Kết luận & điều tôi học được

**Có nên deploy bản fine-tune này không? Không — nhưng không phải vì nó học kém.**

Nó học rất tốt: 0.985 trên tác vụ đích, so với 0.495 của prompt tốt nhất tôi có, và không thua một
mẫu nào trong 50 mẫu target. Nếu hệ thống của tôi chỉ nhận đúng một loại input — ticket CSKH, đi
vào một endpoint chỉ làm triage, không bao giờ nhận gì khác — thì đây là model nên dùng, và cổng
hồi quy có thể bỏ qua một cách có ý thức. Nhưng đó là một giả định về kiến trúc hệ thống, không
phải một kết quả đo được, và nếu giả định đó sai thì thiệt hại là im lặng: model vẫn trả lời, vẫn
đúng định dạng, chỉ là trả lời sai loại câu hỏi. Với năng lực phổ thông còn **0.033**, bất kỳ input
nào lệch khỏi khuôn ticket đều nhận về một object JSON vô nghĩa thay vì một lời từ chối. Một model
hỏng mà vẫn phát ra JSON hợp lệ thì khó phát hiện hơn nhiều so với một model hỏng mà văng lỗi.

**Đâu là đòn bẩy thật?** Xếp theo số đo của chính tôi, ở cùng ngân sách tham số và cùng số step:
**learning rate** (−0.655 khi sai thang) ≫ **vị trí gắn adapter** (−0.050) ≈ **lượng tử hoá**
(−0.055). Rank thì hoàn toàn không có mặt trong danh sách — cả bốn run đều có đúng 10 822 656 tham
số huấn luyện, nên thí nghiệm này đã loại rank ra khỏi phương trình ngay từ thiết kế. Nhưng đòn bẩy
lớn nhất không nằm trong bảng đó: **nó là dữ liệu**. Cả bốn cấu hình đều thừa hưởng cùng một khiếm
khuyết — không có mẫu replay nào — nên cả bốn đều sẽ rớt cổng hồi quy dù tôi vặn LoRA khéo đến đâu.
Cách duy nhất để qua cổng là sửa corpus, không phải sửa siêu tham số.

**Điều lab này thực sự dạy tôi** là kết luận đúng có thể là "chưa nên fine-tune". Trên bài toán này
prompt engineering mua được 0.495 với chi phí bằng không và không gây tổn hại gì; fine-tuning mua
thêm 0.490 nữa nhưng lấy đi 0.633 năng lực phổ thông. Nếu tôi chỉ đo target — cách mà bản lab cũ
chấm bằng perplexity vẫn làm — tôi đã báo cáo một thắng lợi và ship một model đã hỏng. Cổng bốn
nhóm là thứ duy nhất chặn lại, và nó chỉ chặn được vì baseline (b) đã được đóng băng **trước** khi
tôi nhìn thấy bất kỳ kết quả huấn luyện nào.

**Ba điều tôi học được:**

1. **Một prompt tử tế mua được rất nhiều, và mua rẻ.** Baseline (b) đạt 0.495 target với format
   1.000 mà không tốn một step huấn luyện nào — và còn **nhanh gấp 3,5 lần** baseline (a)
   (585 ms so với 2058 ms). Chính xác và chi phí đi cùng chiều chứ không đánh đổi, vì model có
   schema thì phát ra JSON rồi dừng, còn không có schema thì viết văn xuôi tới cạn token. Câu hỏi
   thật không phải "fine-tune có chạy không" mà "nó có đáng so với việc chỉ viết prompt cho tử tế".

2. **Rank không thay thế được vị trí gắn adapter — trên kiến trúc lai thì gần như không thể.**
   Base này xen kẽ linear attention với full attention, và `q_proj`/`v_proj` chỉ tồn tại trên 6
   trong 24 lớp. Để `attn_only` có cùng ngân sách tham số với `correct`, rank phải leo 16 → **271**,
   gấp 17 lần, mà vẫn chỉ chạm được đúng 6 lớp đó. Ngân sách bị dồn cho dày hơn thay vì rộng hơn.

3. **Đo trên lát nhỏ thì sai theo nhiều cách hơn tôi tưởng.** Ở `EVAL_LIMIT=8`, `attn_only` (0.781)
   trông như thắng `correct` (0.719) — chênh lệch đó chỉ bằng nửa mẫu, thuần nhiễu. Cùng lúc,
   latency của baseline (a) đo được 7055 ms ở 8 mẫu nhưng chỉ 2058 ms ở 50 mẫu, vì ~45 giây khởi
   động CUDA bị chia cho 8 thay vì 50. Cùng một run rút gọn làm hỏng **cả** nhóm target lẫn nhóm
   latency, theo hai cơ chế chẳng liên quan gì đến nhau.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 1–5% dữ liệu phổ thông làm replay vào tập huấn luyện
(deck §14.3) rồi train lại `correct` với đúng 58 step, để xem có giữ được điểm target trong khi kéo
nhóm regression về gần mức base không. Đó là biến duy nhất tôi chưa được đụng tới, và là biến trực
tiếp nhắm vào lý do run này rớt.

---

## Phụ lục — thưởng đã làm

- [x] **B1 — NB6 merge + hot-swap.** `results/merge_check.json`: trước merge **0.9850**, sau merge
  **0.9850**, `delta 0.0000` trên đủ 50 mẫu (ngưỡng 0.01). Merge chính xác tuyệt đối — không có
  gì để mất khi gộp `W = W₀ + (α/r)·BA` ở bf16. Hot-swap **3 adapter** (`correct`, `attn_only`,
  `qlora`) trên **một** base đang nạp.

  *Câu hỏi của B1 — merge cho overhead bằng 0, nhưng mất gì?* Mất **tính đa dụng**. Model đã merge
  là một checkpoint cố định: muốn phục vụ khách hàng thứ hai thì phải nạp thêm một bản base đầy đủ
  nữa. Giữ adapter riêng thì base nằm trong VRAM đúng một lần và mỗi tác vụ chỉ tốn vài chục MB —
  đây là toàn bộ lập luận kinh tế của LoRA. Quan sát trực tiếp từ mục 3: cùng một ticket, ba adapter
  cho ba câu trả lời khác nhau (`qlora` đoán `hoan_tien`, hai adapter kia đoán `doi_tra`), chuyển
  qua lại chỉ bằng `set_adapter()` mà không nạp lại gì. **Nên merge khi** bạn phục vụ đúng một tác
  vụ ở lưu lượng cao và muốn đồ thị suy luận giống hệt base; **nên giữ riêng khi** có nhiều tác vụ
  hoặc nhiều khách hàng trên cùng một GPU.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] **B5 — HuggingFace Hub**: <https://huggingface.co/flamehydra/lab21-2A202601156-qwen35-triage-vi>
  Repo public, có `adapters/correct` + toàn bộ `results/` + `submission/REPORT.md`. Model card ghi
  rõ base model là `Qwen/Qwen3.5-0.8B` (tier `CPU`, chạy trên RTX 3050 Ti Laptop 4 GB) và cảnh báo
  không deploy adapter này cho lưu lượng hỗn hợp vì nhóm regression chỉ còn 0.033.

  Mã nguồn: <https://github.com/nguyengochiep/K4-Track3-DAY21-2A202601156-NguyenNgocHiep>

---

## Phụ lục — ghi chú môi trường

Ba trục trặc gặp phải khi chạy lab trên Windows 11 + RTX 3050 Ti 4 GB, ghi lại để tái lập:

1. **`colab_run.py` truyền `PYTHONUNBUFFERED` cho tiến trình con nhưng không truyền
   `PYTHONIOENCODING`.** Notebook in tiếng Việt, stdout là pipe → Windows chọn cp1252 → NB1 chết
   với `UnicodeEncodeError` trước khi tokenize được gì. Đã sửa tại `scripts/colab_run.py:58`.
2. **`torchao 0.18` gọi `torch.nn.functional.ScalingType`**, API chỉ có từ torch 2.13; với torch
   2.9.1+cu126 thì import transformers chết. Pin `torchao>=0.16` trong `requirements.txt` tồn tại
   để đè bản 0.10.0 mà **Colab** cài sẵn, không cần thiết trên máy local — gỡ torchao là xong,
   import đó được `is_torchao_available()` bảo vệ.
3. **`verify.py` báo FAIL "eval sets unmodified" dù không đụng vào file nào.** Nguyên nhân:
   `core.autocrlf=true` khiến git đổi LF→CRLF lúc checkout, nên sha256 của byte thô khác
   `checksums.json`. Chuẩn hoá CRLF→LF thì sha trùng khít cả bốn file, và `git status` sạch —
   nội dung nguyên vẹn, chỉ khác ký tự xuống dòng. Đã sửa `_sha()` để hash **nội dung** thay vì
   quy ước xuống dòng của hệ điều hành.

4. **NB6 chết ở mục hot-swap trên GPU 4 GB.** `del merged` không đủ: `merge_and_unload()` trả về
   base đã gộp nhưng biến `model` vẫn giữ `PeftModel` cũ, nên khi `load_base()` được gọi lần hai
   thì hai model cùng nằm trong VRAM (biến chỉ được gán lại *sau khi* hàm trả về).
   `device_map="auto"` phải offload layer 4–23 sang CPU và PEFT từ chối vì thiếu `offload_dir`.
   Đúng là lỗi mà `HARDWARE-GUIDE.md` liệt kê ở bảng sự cố — *"OOM ở run thứ 2: model cũ chưa được
   giải phóng"* — xảy ra ngay trong notebook dạy điều đó. Sửa: `del merged, model`.

Ba trong bốn lỗi trên chỉ xuất hiện ngoài môi trường tham chiếu (Colab T4 / Linux, GPU ≥ 12 GB).
Lỗi thứ tư chỉ xuất hiện dưới 12 GB VRAM. Không lỗi nào ảnh hưởng tới con số trong báo cáo này.
