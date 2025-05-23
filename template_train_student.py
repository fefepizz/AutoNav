from models.cnn_student import StudentCNN
from models.cnn_teacher import TeacherCNN
from models.distillation import DistillationLoss

# Inside training loop
student_model = StudentCNN().to(device)
teacher_model = TeacherCNN().to(device)
teacher_model.load_state_dict(torch.load("path/to/teacher_weights.pth"))
teacher_model.eval()

criterion = DistillationLoss(temperature=4.0, alpha=0.7)
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)

# In training loop
for inputs, labels in dataloader:
    inputs, labels = inputs.to(device), labels.to(device)

    with torch.no_grad():
        teacher_outputs = teacher_model(inputs)

    student_outputs = student_model(inputs)
    loss = criterion(student_outputs, teacher_outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
