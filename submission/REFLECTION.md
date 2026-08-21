# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Prompt tối ưu vừa chính xác hơn vừa **rẻ hơn**: 0.495 so với 0.000 ở target, đồng thời nhanh gấp
3,5 lần (585 ms so với 2058 ms). Tôi vẫn nghĩ chất lượng và chi phí là thứ phải đánh đổi. Ở đây
chúng đi cùng chiều, vì nguyên nhân của cả hai là một: model biết schema thì phát ra JSON rồi dừng,
không biết thì viết văn xuôi cho tới cạn 160 token.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

NB4 — ba run huấn luyện đối chứng — chiếm khoảng một nửa ngân sách. Không đúng dự đoán: tôi tưởng
phần sinh văn bản mới là phần đắt, vì nó phải chạy ba lượt trên toàn tập eval. Hoá ra `EVAL_LIMIT`
không rút ngắn được NB4 chút nào, còn phần sinh thì rẻ hơn tôi tưởng khi tập đủ lớn để chia đều
chi phí khởi động.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Rằng chất lượng LoRA chủ yếu là chuyện chọn rank. Bốn run trong lab đều có **cùng số tham số huấn
luyện** (10 822 656) và cùng số step, nên rank bị loại khỏi phương trình — thứ tạo ra khác biệt là
*gắn adapter ở đâu* và *learning rate ở thang nào*. Run `wrong_lr` chỉ đổi một con số và mất sạch
cả điểm format.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Dùng để dựng môi trường (torch CUDA, xung đột torchao) và chẩn đoán lỗi. Chỗ nó sai rõ nhất là
**ước lượng thời gian**: ban đầu nói full run mất 75–100 phút, sau khi có s/step đầu tiên đổi thành
2 giờ 20, cuối cùng số đo thật lại nằm giữa. Run smoke cũng được đoán là 15–25 phút nhưng chạy hết
48 phút. Bài học: các con số ngoại suy từ tỷ lệ phần cứng không đáng tin, phải đo mới biết — đúng
tinh thần của chính lab này.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Viết một prompt thật tử tế cho model gốc, đo nó trên tập eval, rồi **đóng băng** cả tập eval lẫn
prompt đó trước khi train bất cứ thứ gì. Nếu không có mốc đó thì không có cách nào biết fine-tune
có đáng tiền hay không — và nếu đo sau, tôi sẽ vô thức chỉnh mốc cho tới khi bản fine-tune trông
thắng.
