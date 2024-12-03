def get_recommendation(age):
    """
    Takes an age as input and returns a recommendation based on the age range.
    """
    # Define recommendations based on age ranges
    recommendations = {
        (0, 12): "LEGO Building Set - Creative Play for Kids",
        (13, 18): "Young Adult Fiction Book - 'The Hunger Games'",
        (19, 20): "Noise-Cancelling Headphones - Perfect for College Students",
        (21, 22): "Your potential is limitless—go beyond the ", age, "nd mile!",
        (23, 24): "You've got ",age," hours in a day to chase your dreams—make them count!",    
        (25, 26):"Success is built step by step—start your", age,"th step today!",
        (27, 28): "Life gives us", age, " fresh chances every day—seize this one!",
        (29, 30): "You're ", age, " times stronger than you think—believe in yourself!",
        (31, 32): "Even the longest journey begins with the first ", age, " seconds of courage.",
        (33, 34): "You are just ", age, " decisions away from a breakthrough—choose wisely!",
        (35, 36): "In ", age, " days, you can build a habit that transforms your life!",
        (37, 38): "Don't let the ", age, "th challenge stop you—you're almost there!",
        (39, 40): "You have ", age, " reasons to be grateful today—focus on them!",
        (41, 45): "Fitness Tracker - Stay Active and Healthy",
        (46, 49): "Cookware Set - For Gourmet Home Cooking",
        (50, 64): "Gardening Kit - Cultivate Your Green Thumb",
        (65, 120): "Travel Guide - Plan Your Next Adventure",
    }
    
    # Find the correct recommendation for the given age
    for age_range, item in recommendations.items():
        if age_range[0] <= age <= age_range[1]:
            return item
    
    # Handle invalid age inputs
    return "Sorry, we couldn't find a recommendation for this age."
