import matplotlib.pyplot as plt
from ultralytics import YOLO

def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 5
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
    print(f"{num_freeze} layers are freezed.")

if __name__ == '__main__':
    # Initialize the model
    model = YOLO("model_box-counting-YOLOv8n_1280_v4.pt")

    # Add callback to freeze layers
    model.add_callback("on_train_start", freeze_layer)

    # Train the model
    results = model.train(data="dataset/data.yaml", epochs=10, imgsz=1280,batch=8)

    # Validate the model after training
    model.val(data="dataset/data.yaml")

    # Plot training loss
    fig, ax = plt.subplots()
    results.plot(ax=ax)

    # Save the plot
    fig.savefig("training_loss_plot.png")

    # Save the model after validation
    model.save("model_box-counting-YOLOv8n_1280_v4_trained.pt")
