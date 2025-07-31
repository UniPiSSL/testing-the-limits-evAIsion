def pick_model():
    print("Choose a model to use:")
    print("1. Fully Connected Neural Network (FCNN)")
    print("2. LeNet (Convolutional Neural Network)")
    print("3. Simple CNN")
    print("4. MobileNetV2")
    print("5. VGG11")
    try:
        choice = int(input("Enter your choice (1-5): "))
        model_options = {
            1: 'fcnn',
            2: 'lenet',
            3: 'simple_cnn',
            4: 'mobilenetv2',
            5: 'vgg11'
        }
        return model_options.get(choice, 'fcnn')
    except ValueError:
        print("Invalid input. Defaulting to FCNN.")
        return 'fcnn'

def pick_attack():
    print("Choose an attack:")
    print("1. FGSM")
    print("2. PGD")
    print("3. DeepFool")
    print("4. Carlini & Wagner")
    try:
        choice = int(input("Enter your choice (1-4): "))
        if choice not in [1, 2, 3, 4]:
            raise ValueError("Choice must be between 1 and 4.")
        return choice
    except ValueError as e:
        print(f"Invalid input: {e}. Exiting.")
        exit()