def get_recommendation(age):
    """
    Takes an age as input and returns a recommendation based on the age range.
    """
    # Define recommendations based on age ranges
    recommendations = {
        (0, 12): "LEGO Building Set - Creative Play for Kids",
        (13, 18): "Young Adult Fiction Book - 'The Hunger Games'",
        (19, 20): "Noise-Cancelling Headphones - Perfect for College Students",
        (21, 22): f"Your potential is limitless—go beyond the {age}nd mile!",
        (23, 24): f"You've got {age} hours in a day to chase your dreams—make them count!",
        (25, 26): f"Success is built step by step—start your {age}th step today!",
        (27, 28): f"Life gives us {age} fresh chances every day—seize this one!",
        (29, 30): f"You're {age} times stronger than you think—believe in yourself!",
        (31, 32): f"Even the longest journey begins with the first {age} seconds of courage.",
        (33, 34): f"You are just {age} decisions away from a breakthrough—choose wisely!",
        (35, 36): f"In {age} days, you can build a habit that transforms your life!",
        (37, 38): f"Don't let the {age}th challenge stop you—you're almost there!",
        (39, 40): f"You have {age} reasons to be grateful today—focus on them!",
        (41, 45): f"Fitness Tracker - Stay Active and Healthy at {age}",
        (46, 49): f"{age} Cookware Set - For Gourmet Home Cooking",
        (50, 64): f"Gardening Kit - Cultivate Your {age} Green Thumb",
        (65, 120): f"Travel Guide - Plan Your Next {age} Adventure",
    }
    
    # Find the correct recommendation for the given age
    for age_range, item in recommendations.items():
        if age_range[0] <= age <= age_range[1]:
            return item
    
    # Handle invalid age inputs
    return "Sorry, we couldn't find a recommendation for this age."
