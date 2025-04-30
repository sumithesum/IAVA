import torch
import csv

def generate_submission(model, test_loader, index_to_landmark, device, output_path="submission.csv"):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for img_id, pred_idx in zip(image_ids, preds):
                landmark_id = index_to_landmark[pred_idx]
                predictions.append((img_id, landmark_id))
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "landmark_id"])
        writer.writerows(predictions)

    print(f"Submission salvat ca {output_path} ")