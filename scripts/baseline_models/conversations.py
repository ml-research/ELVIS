# Created by MacBook Pro at 17.07.25

def deepseek_conversation(train_positive, train_negative, principle):
    conversation = [
        {
            "role": "<|User|>",
            "content": "Describe the difference between these images.",
            "images": [str(img) for img in train_positive + train_negative],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    return conversation
def llava_conversation(train_positive, train_negative, principle):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_positive[0]},
                {"type": "text", "text": f"You are an AI reasoning about visual patterns based on Gestalt principles.\n"
                                         f"Principle: {principle}\n\n"
                                         f"We have a set of images labeled Positive and a set labeled Negative.\n"
                                         f"You will see each image one by one.\n"
                                         f"Describe each image, note any pattern features, and keep track of insights.\n"
                                         f"After seeing all images, we will derive the logic that differentiates Positive from Negative. "
                                         f"The first positive image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_positive[1]},
                {"type": "text", "text": f"The second positive image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_positive[2]},
                {"type": "text", "text": f"The third positive image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_negative[0]},
                {"type": "text", "text": f"The first negative image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_negative[1]},
                {"type": "text", "text": f"The second negative image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_negative[2]},
                {"type": "text", "text": f"The third negative image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Now we have seen all the Positive and Negative examples. "
                                         "Please state the logic/rule that distinguishes them. "
                                         "Focus on the Gestalt principle of "
                                         f"{principle}."},
            ],
        },

    ]
    return conversation

def llava_eval_conversation(image, logic_rules):
    conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",
                     "text": f"Using the following reasoning rules: {logic_rules}. "
                             f"Classify this image as Positive or Negative."
                             f"Only answer with positive or negative."},
                ]
            }
        ]
    return conversation