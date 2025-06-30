import pandas as pd
import os


# Parameter, change dataset for testing
#dataset = "mwoz"
dataset = "sgd"

key_choice = "public"

import sys
name = sys.argv[1]
dataset = sys.argv[2]


# Booking / Informational Request

post_fix = 'full_eff3_5fewshot_V3_deter'

if name == "claude":
    prompt_id = "2000claude"
    model_choice = 'claude-3-7-sonnet-latest'
elif name == "deepseek":
    prompt_id = "2000deepseek"
    model_choice = 'ark-deepseek-r1-250120'
elif name == 'gpt4o':
    model_choice = 'gpt-4.1'
    prompt_id = "2000gpt4o"
elif name == "gemini":
    prompt_id = "gemini"
    model_choice = 'gemini-2.5-pro-preview-05-06'


openAIport = False



prompt_id = prompt_id + "-" + post_fix
output_dir = "outputs/gpt_response_" + dataset + "_gptresult" + prompt_id + "/" 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#model_choice = 'mistral-large-latest'


import shutil
shutil.copyfile(__file__, output_dir +"prompt.py")


if dataset == "mwoz":
    data_dir = "MWOZ.txt"
elif dataset == "sgd":
    data_dir = "SGD.txt"
elif dataset == "jddc":
    data_dir = "JDDC.txt"
elif dataset == "redial":
    data_dir = "ReDial.txt"

#data_dir = "SGD.txt"
#data_dir = "ReDial.txt"
#data_dir = "JDDC.txt"
#mode = "dialogue-based"


def main():
    data = {
        'dialogue': [],
        'dia_act': [],
        'dia_sat': [],
    }

    # Read data from file
    with open(data_dir) as f:
        dialogue = []
        dia_act = []
        dia_sat = []
        for line in f:
            if line == '\n':
                data['dialogue'].append(dialogue)
                data['dia_sat'].append(dia_sat)
                dialogue = []
                dia_sat = []
            else:
                line = line.split("\t")
                dialogue.append(line[0] + '\t' + line[1])
                dia_sat.append(line[3].strip("\n"))

    # change format of dialogue
    def format_dialogue(dia_list):
        dia_format = []
        for dia_turn in dia_list:
            dia_turn_list = dia_turn.split("\t")
            dia_format.append(dia_turn_list[0] + ": " + dia_turn_list[1])
        return dia_format
    format_dia = []
    for dia_list in data["dialogue"]:
        format_dia.append(format_dialogue(dia_list))

    import numpy as np
    def get_main_score(scores):
        number = [0,0,0,0,0]
        for item in scores:
            number[int(item)-1] += 1
        score = np.argmax(number)
        return score
    
    # Dictionay that map original label to label of 0, 1, 2
    dic_lab = {
        4:2,
        3:2,
        2:1,
        1:0,
        0:0
    }
    # Label list for each sample in dataset.
    lab_list = []
    for dia_sat in data['dia_sat']:
        sat_last_turn = dia_sat[-1]
        lab = dic_lab[get_main_score(sat_last_turn.split(","))]
        lab_list.append(lab)
    data["dia_sat"] = lab_list


    nat_sample_id = []
    sat_sample_id = []
    dissat_sample_id = []

    for i, lab in enumerate(data["dia_sat"]):
        if lab == 0:
            dissat_sample_id.append(i)
        elif lab == 1:
            nat_sample_id.append(i)
        elif lab == 2:
            sat_sample_id.append(i)

    data_test = {}
    data_test["dialogue"] = format_dia

    # Index of dialogue sample for testing
    sample_idx_list = []
    for subdir, dirs, files, in os.walk(output_dir):
        for file in files:
            if file.endswith(".txt"):
                sample_idx = int(file.split(".txt")[0])
                sample_idx_list.append(sample_idx)

    #MWOZ sample ID
    if dataset == 'mwoz':
        test_sample_id = [518, 9, 522, 528, 26, 33, 34, 41, 554, 45, 571, 63, 68, 581, 582, 583, 593, 598, 93, 95, 609, 620, 118, 637, 638, 130, 133, 138, 657, 153, 156, 158, 671, 168, 169, 177, 696, 186, 187, 699, 701, 194, 196, 710, 211, 729, 745, 241, 254, 772, 773, 775, 267, 268, 271, 798, 286, 288, 809, 299, 815, 305, 820, 314, 323, 842, 845, 853, 345, 875, 877, 378, 382, 383, 900, 392, 393, 911, 917, 922, 415, 928, 927, 931, 422, 936, 425, 429, 434, 951, 959, 449, 969, 462, 471, 983, 476, 483, 995, 499, 508]
        valid_sample_id = [516, 6, 7, 11, 530, 18, 532, 539, 553, 47, 560, 566, 60, 67, 71, 597, 97, 624, 115, 628, 120, 121, 128, 132, 653, 668, 672, 163, 678, 170, 691, 180, 705, 202, 714, 716, 204, 209, 735, 737, 229, 234, 749, 240, 759, 770, 264, 270, 783, 274, 275, 278, 793, 287, 801, 802, 291, 292, 297, 810, 299, 308, 317, 330, 338, 339, 854, 342, 856, 344, 859, 864, 359, 364, 878, 881, 375, 888, 380, 896, 387, 912, 410, 428, 443, 444, 963, 965, 967, 459, 463, 977, 981, 472, 988, 994, 487, 489, 497, 506, 509]
        print("MWOZ samples")

    if dataset == 'sgd':
    #SGD sample ID
        test_sample_id = [472, 669, 341, 517, 194, 993, 173, 480, 999, 672, 923, 898, 207, 318, 865, 227, 955, 284, 877, 821, 982, 275, 846, 631, 149, 364, 92, 375, 41, 256, 271, 483, 547, 828, 58, 856, 216, 107, 972, 721, 778, 896, 957, 166, 116, 434, 857, 28, 303, 117, 946, 399, 609, 605, 676, 492, 602, 22, 323, 737, 29, 431, 575, 442, 482, 13, 663, 766, 44, 520, 277, 338, 282, 722, 624, 124, 607, 748, 75, 439, 447, 858, 419, 930, 127, 916, 224, 801, 72, 103, 357, 479, 423, 285, 210, 84, 350, 135, 992, 228]
        valid_sample_id = [526, 531, 19, 563, 571, 62, 63, 577, 578, 580, 589, 81, 83, 604, 94, 618, 620, 112, 627, 115, 634, 130, 653, 664, 152, 673, 675, 681, 683, 689, 181, 694, 182, 184, 697, 701, 706, 196, 716, 204, 209, 729, 732, 226, 745, 235, 757, 758, 759, 764, 772, 773, 261, 780, 784, 785, 788, 795, 797, 798, 799, 290, 805, 808, 311, 314, 830, 320, 842, 847, 849, 851, 853, 342, 862, 869, 358, 377, 384, 388, 396, 400, 914, 918, 921, 928, 418, 933, 950, 964, 968, 457, 970, 460, 461, 470, 998, 501, 505, 507]
        print("SGD samples")

    #test_sample_id = [158, 508, 598]
    from openai import OpenAI
    test_sample_id = [i for i in range(100)]
    idx_except = 0
    fail_idx = []

    overlap = False
    for i in range(1001):
        print('idx:' + str(i))
        if overlap == False:
            if i in sample_idx_list:
                print("idx found, skipping")
                continue
        dialogue = ""
        for sentence in data_test["dialogue"][i]:
            if "OVERALL" in sentence:
                continue
            dialogue += sentence + "\n"

        # Prompt
        system_prompt = """You are a User Satisfaction Estimation Classifier that evaluates user satisfaction in a dialogue through four steps.
        **Step 1: Extract Main Goals**
        Identify the user's main goals from the dialogue, 
            such as booking train tickets, reserving tables, booking hotel rooms, asking for recommendations, looking for information, etc.
        And then, combine related goals into one main goals.
        Example of combining goals includes but not limited to:
            1. "Seek for recommendation of restaurant" and "Make reservation of the restuarant" can be combined into "Find and Make reservation"
            2. "Get information such as Wifi availability of a restaurant" and "Make reservation of the restuarnt" can be combined into one goal because they are related.
            3. Making adjustment of goals, such as intially seek for a certain type of restuarant, but there are no options so the user changes to another type can be combined into one goal. 

        **Step 2: Dialogue Efficiency Classification
        - For each goals, evaluate if the system handle the goals efficiently.
        - Classify them into efficient and inefficient
            1) Efficient: The system does not take unnecessary turns due to system failure or system misunderstanding users.
            2) Inefficient: The system takes unnecessary turn to handle user's requests due to system's error, failure in handle user's initial requests, or system misunderstanding users. 

        **Step 3: Goals fulfillment classification**
        - For each goal, classify it as positive, neutral, or negative based on the following criteria:
        1. Positive if all the following criteria is met
            1) system fulfills the main goal or alternative goals when intial goal cannot be fulfilled
            2) system handles the goals efficiently according to step 2
        2. Neutral if the following is met:
            1) The system fulfills the main goal or its alternative but system does not handles the goals efficiently according to step 2.
        3. Negative if the system is unable to fulfill the main goal such as providing wrong information.
    
        **Step 4: Evaluate User Satisfaction Rate**
        - According to the step 3, assign a task fulfillment rate score from 1 to 3 based on the classification of goals:
        1) If all goals in are classified as "Negative", classified it as score of 1.
        2) If goals in step 3 are classified as neutrals, or mixture of positive, neutral, or negative, classified it as score of 2.
        3) If all the goals in step 3 are classified as positive without any goals classified as neutral or negative, clssified it as score of 3.
        
        Here is 5 examples of a dialogue and answer format for step 1 to step 4:
        """
            
        if dataset == "sgd":
            system_prompt = system_prompt + """
                Example 1:
                USER	What is the weather like on the March 4th?
                SYSTEM	In which city should I look?
                USER	The weather in Mill Vallley.
                SYSTEM	The weather should be around 90 degrees and there is a 7 percent chance of rain.	
                USER	How humid will the temperature be?
                SYSTEM	There is about a 18 percent humidity.
                USER	Could you find a song from Jessie J album Who You Are?
                SYSTEM	There is 1 song Mamma Knows Best by Jessie J from the album Who you are.
                USER	Yes, play it on the kitchen device.
                SYSTEM	Play Mamma knows best on the kitchen speaker.
                USER	No, play it on bedroom speaker instead.
                SYSTEM	Play Mamma Knows Best on bedroom speaker.
                USER	Yes, thank you. What is the genre of the song?
                SYSTEM	The genre of the song is pop.
                USER	Thank you for your help.
                SYSTEM	Can I help with anything else?
                USER	No, that is all.
                SYSTEM	Have a good day.


                ### Step 1: Extract all of the main goals of the user from the dialogue.
                1. Find information about weather and temperature.(Find information about weather, and Find information about temperature are combined because they are related)
                2. Find a song and play it on the designated area
                3. Get the genre of the song

                ### Step 2: For each goal, Classify the efficiency of system handling the goal.
                1. **Goal: Find information about weather
                - Efficiency: Efficient
                - Reason: The system quickly provided the requested weather information (temperature and chance of rain) and additional humidity details without unnecessary turns.

                2. **Goal:  Find a song and play it on the designated area
                - Efficiency: Efficient
                - Reason: The system fulfilled the main goal efficiently, including adjusting the playback device as corrected by the user.

                3. **Goal:  Get the genre of the song
                - Efficiency: Efficient
                - Reason: The system provided the information efficiently

                ### Step 3: For each goal, classify it as positive, negative, or neutral.
                1. **Get weather information for Mill Valley on March 4th:**
                - Classification: Positive
                - Reason: The system fulfilled the main goal efficiently and provided additional requested details (humidity).

                2. **Find and play a song from Jessie J's album "Who You Are."**
                - Classification: Positive
                - Reason: The system fulfilled the main goal efficiently, including adjusting the playback device as corrected by the user.

                3. **Get the genre of the song.**
                - Classification: Positive
                - Reason: The system fulfilled the main goal successfully and efficiently.

                ### Step 4: User Satisfaction: 3


                Example 2:
                USER	I'd like to check out a film.
                SYSTEM	Where would you like to see one?
                USER	In San Ramon.
                SYSTEM	I got 9 for you. Breakthrough, Captain Marvel, and Dumbo, to start out.
                USER	Let go with Breakthrough. See if you can find an IMAX at Regal Crow Canyon on the 9th.
                SYSTEM	At Regal Cinemas Crow Canyon 6? There is a 4 pm show.
                USER	That is cool.
                SYSTEM	So should I pick you up some tickets?	
                USER	No. Don't do that.
                SYSTEM	What should I do?
                USER	Make a reservation at a nearby restaurant.
                SYSTEM	Do you have one in mind.
                USER	Yeah, I like McDonald's.
                SYSTEM	What time is good for you?
                USER	Half past 1 in the afternoon.
                SYSTEM	Table for 2 at McDonald's in San Ramon on March 9th at 1:30 pm. Is that accurate?
                USER	No, I need it for 3 at 11:45 am.
                SYSTEM	Table for 3 at 11:45 am?
                USER	Yes, thats right.
                SYSTEM	Sorry, can't do that at McDonald's. Should I try 12 pm on MArch 9th for 3?	
                USER	Yes, try that.
                SYSTEM	Done. You have a table.
                USER	Great work. Thanks.
                SYSTEM	What else do you have for me.	
                USER	Nothing. Just relax.
                SYSTEM	Will do, thanks.


                ### Step 1: Extract all of the main goals of the user from the dialogue.
                1. Find information about a film ("Breakthrough") 
                2. Make a reservation at a nearby restaurant (McDonald's) 

                ### Step 2: For each goal, list all the related requirements specified by the user.
                1. **Goal: Find information about a film ("Breakthrough") **
                - Efficiency: Efficient
                - Reason: The system quickly provided the available showtime (4 pm) at the requested theater without unnecessary turns.

                2. **Goal: Make a reservation at a nearby restaurant (McDonald's)**
                - Efficiency: Inefficient
                - Reason: The system successfully make reservation, but initially fail for 11:45 am, making it inefficient.

                ### Step 3: For each goal, classify it as positive, negative, or neutral.
                1. **Find information about a film ("Breakthrough"):**
                - Classification: Positive
                - Reason: The system fulfilled the goal efficiently and succesfully.

                2. **Make a reservation at a nearby restaurant (McDonald's):**
                - Classification: Neutral
                - Reason: The system fulfilled the goal succesfully but not efficiently.

                ### Step 4: User Satisfaction: 2
                Reason: One goal was classified as positive, and the other was classified as neutral.


                Example 3:
                USER	I'd like to find a place to eat.
                SYSTEM	In which city? What kind of food would you like?
                USER	I'd like Sichuan in San Jose.
                SYSTEM	Chef Li in San Jose is nice.
                USER	What's their address?
                SYSTEM	2033 Camden Avenue # F3.
                USER	Can you check on another? I'm looking for something that is intermediate priced.
                SYSTEM	I found 9 possibilities. China Delight in San Jose is suppose to be nice.
                USER	Do they serve alcohol?
                SYSTEM	They do not.
                USER	Okay that sounds good.
                SYSTEM	Would you like a reservation?
                USER	Not just now.
                SYSTEM	Is there anything else you need?	
                USER	I'd like to find a movie nearby.
                SYSTEM	I found 10 nearby movies. What about A Madea Family Funeral, After or Breakthrough?	
                USER	What are some others? I'd like a 3d movie.
                SYSTEM	There's How to Train Your Dragon: The Hidden World, Little or Missing Link.	
                USER	How to Train Your Dragon would be great.
                SYSTEM	Shall I order tickets?
                USER	No, not just yet.
                SYSTEM	Is there anything else you need?	
                USER	Let's go ahead and make a reservation at the restaurant.
                SYSTEM	For what time?
                USER	Evening 7:30 would be good.
                SYSTEM	Please verify: You'd like a reservation for 2 at China Delight in San Jose at 7:30 pm on March 1st.
                USER	Correct. Do they have live music?
                SYSTEM	Sorry I couldn't make the reservation. Is there anything else you need?	
                USER	Try it again for 5:30 in the evening.
                SYSTEM	To verify: You would like a reservation for 2 at China Delight in San Jose at 5:30 pm on March 1st.	
                USER	That is right.
                SYSTEM	You're reservation is confirmed.	
                USER	Thank you. That's all I need for now.
                SYSTEM	Have a great day.


                ### Step 1: Extract all of the main goals of the user from the dialogue.
                1. Find a restaurant and make reservation.(Find the restaurant and make reservation are combined because they are related)
                2. Find a 3D movie nearby.


                ### Step 2: For each goal, list all the related requirements specified by the user.
                1. **Goal: Find a restaurant serving Sichuan food in San Jose and make reservation.**
                - Efficiency: InEfficient
                - Reason: The systems initially fail to provides option with intermediate prices and had to take addtionally turn to provide other option. Additionally, the system initially failed to make the reservation and required the user to re-specify the time.

                2. **Goal: Find a 3D movie nearby.**
                - Efficiency: InEfficient
                - Reason: The system provided multiple options and take additionally turn to adjusted based on the user's request for a 3D movie.


                ### Step 3: For each goal, classify it as positive, negative, or neutral.
                1. **Find a restaurant serving Sichuan food in San Jose:**
                - Classification: Neutral
                - Reason: The system eventually fulfilled the goal but initially failed to make the reservation, requiring additional turn.

                2. **Find a 3D movie nearby:**
                - Classification: Neutral
                - Reason: The system fulfilled the main goal efficiently by providing suitable options and adjusting to the user's request for a 3D movie.

                ### Step 4: User Satisfaction: 2
                Reason: Two goals were classified as neutral, resulting in a mixture of positive and neutral classifications.


                Example 4:
                USER	I would like to reserve event tickets.
                SYSTEM	Do you have a date and event in mind? How many tickets?		
                USER	I want two tickets to the Lizzo event next Thursday.
                SYSTEM	For what city?	
                USER	I'm searching for events around SD.	
                SYSTEM	Please confirm the following details: You have 2 tickets to Lizzo next Thursday in San Diego.
                USER	No, I want 1 ticket to an event on Wednesday next week.
                SYSTEM	Please confirm the following details: You have 1 ticket for an event next Wednesday.
                USER	Yes, that works.
                SYSTEM	Sorry, but I could not reserve the tickets for you. Is there anything else I can help you with?
                USER	Could you try again? I am looking for tickets to Nationals Vs Rockies around the District of Columbia.
                SYSTEM	Please confirm the following details: You have 1 ticket in your reservation for Nationals Vs Rockies next Wednesday in Washington D.C.
                USER	Yes, that works.
                SYSTEM	Sorry, I could not reserve the tickets for you. Is there anything else I can help you with?	
                USER	No, thank you very much for your help.
                SYSTEM	Have a great day.

                ### Step 1: Extract all of the main goals of the user from the dialogue.
                1. Reserve event tickets for an event

                ### Step 2: For each goal, list all the related requirements specified by the user.
                1. **Goal: Reserve event tickets.**
                - Efficiency: Inefficient
                - Reason: System fails to make reservation efficiently by requiring multiple attempts for the reservation.

                ### Step 3: For each goal, classify it as positive, negative, or neutral.
                1. **Reserve event tickets:**
                - Classification: Negative
                - Reason: The system failed to reserve the tickets.

                ### Step 4: User Satisfaction: 1
                Reason: All goals in step 3 are classified as negative. The system was unable to fulfill the user's main goal.


                Example 5:
                USER	I have a toothache, find a dentist please.
                SYSTEM	Sure, Which city would you prefer a dentist in?	
                USER	SFO please
                SYSTEM	How about Balboa Dental Care located in San Francisco.
                USER	Maybe, what's the phone number?	
                SYSTEM	The number is 415-387-8802
                USER	More options please.
                SYSTEM	Sure, How about Lee Myoung located also in San Francisco?
                USER	Find out if they can do cosmetic services and I will need their phone number as well.
                SYSTEM	Unfortunately, they do not offer cosmetic services, I show a number for 415-239-4140.
                USER	Ok, I would like to book with this one.
                SYSTEM	Would you like my help booking this appointment?
                USER	Yes, please book for me.
                SYSTEM	What time is best for you?
                USER	Find out if I have anything available at 16:30?
                SYSTEM	Which date would work best for your booking?
                USER	The 8th of this month would be the best day.
                SYSTEM	please confirm a booking for Lee Myoung on March 8th at 4:30 pm. Is that right?
                USER	Yes, perfect.
                SYSTEM	Great news, I was able to confirm your booking.
                USER	Great, let's go ahead and add this to my calendar please.
                SYSTEM	Sure, I can do that for you. What would you like to call this event and at what time should I add this?
                USER	Let's label it not at work and put it at 4:15 in the evening please.
                SYSTEM	sure thing, will this event be happening at 90 Woodacre Drive # 108.
                USER	Yeah, that is right.
                SYSTEM	Great, please confirm we are adding the event titled not at work for March 8th at 4:15 pm. The address is 90 Woodacre Drive # 108. Do I have this correct?
                USER	Yep, sounds right.
                SYSTEM	Great, I was able to update your calendar successfully.
                USER	Thank you for your help. Good bye.
                SYSTEM	Your welcome, Good bye.


                ### Step 1: Extract all of the main goals of the user from the dialogue.
                1. Find and book an appointment with a dentist in San Francisco.(Find a dentist and book the appointment are combined because they are related)
                2. Add the appointment to the user's calendar.

                ### Step 2: For each goal, list all the related requirements specified by the user.
                1. **Goal: Find and book an appointment with a dentist.**
                - Efficiency: InEfficient
                - Reason: The system had to take additional turns to provide other options

                2. **Goal: Add the appointment to the user's calendar.**
                - Efficiency: Efficient
                - Reason: The system accurately confirmed the event details and successfully added it to the calendar.

                ### Step 3: For each goal, classify it as positive, negative, or neutral.
                1. **Find and Book an appointment with a dentist:**
                - Classification: Neutral
                - Reason: The user had to take additional turns to seek for more options although the system successfully booking the appointment.

                2. **Add the appointment to the user's calendar:**
                - Classification: Positive
                - Reason: The system fulfilled the main goal efficiently by accurately confirming and adding the event to the calendar.

                ### Step 4: User Satisfaction: 2
                Reason: One goal in step 3 is classified as positive, and the other is classified as neutral.

                """
        elif dataset == "mwoz":
            system_prompt = system_prompt + """
            Example 1: 
            USER	I'm looking for Korean food in the east area.
            SYSTEM	There are not any Korean restaurants in the East. Would you like me to look in a different area of the city?	
            USER	How about Indian food?
            SYSTEM	Yes there are four different restaurants. What is your price range? There's moderate to expensive available.	
            USER	It does not matter, the price is whatever. Just book me one for Saturday.
            SYSTEM	What time do you want me to book for?
            USER	I need a table for 5 people at 14:30. Could you give me a reference number?
            SYSTEM	Curry Prince is highly recommended. I booked a table for 5 at 14:30 on saturday, your reference number is FZLK0ZCF. Can I help with anything else?
            USER	Does Curry Prince accept to-go orders?
            SYSTEM	Sorry, they don't have the capacity for take-out at this time. Do you have any other questions for us?		
            USER	Does the restaurant have vegetarian options?
            SYSTEM	There is no vegetarian menu at Curry Prince. Would you like any more assistance?		
            USER	Does this restaurant have gluten free options?
            SYSTEM	There are no gluten free options and Curry Prince. Do you have any other things I can help you with today?		
            USER	yeah i need info on stuff to do in town.
            SYSTEM	Is there a specific type of attraction you're looking for?	
            USER	Yes, I need the address for a swimming pool please.
            SYSTEM	My favorite pool is Parkside, in the center of town.
            USER	That would be fine can I get the address please?
            SYSTEM	Yes, the address is Gonville Place. Is there anything else I can help with?
            USER	No that's all I need for today. Bye.


            ### Step 1: Extract all of the main goals of the user from the dialogue.
            1. **Find and make a reservation at a restaurant (initially Korean, then Indian).**
            2. **Get information about activities in town.**

            ### Step 2: For each goal, Classify the efficiency of system handling the goal.
            1. **Goal: Find and make a reservation at a restaurant.**
            - Efficiency: InEfficient
            - Reason: The system initially fail to find Korean restraurants and had to take addtional turns to adjust to Indian restaurant.

            2. **Goal: Get information about activities in town.**
            - Efficiency: Efficient
            - Reason: The system directly provides the address of the swimming pool after the user specifies their need, without any misunderstandings or delays.

            ### Step 3: For each goal, classify it as positive, negative, or neutral.
            1. **Find and make a reservation at a restaurant:**
            - Classification: Neutral
            - Reason: The system efficiently handles the reservation efficiently and provides information user requested (to-go orders, vegetarian options, gluten-free options), which affects overall satisfaction.

            2. **Get information about activities in town:**
            - Classification: Positive
            - Reason: The system fulfills the user's request efficiently and completely.

            ### Step 4: User Satisfaction: 2
            - Reason: One goal is classified as neutral and one goal is classified as positive


            Example 2:
            USER	I'm seeking an attraction! A boat one will be nice.
            SYSTEM	Great! I have four for you. Two are in the centre. Any location you'd like in particular?
            USER	No, I'd like the name of one in the centre please.
            SYSTEM	Sure, I can get tickets for the cambridge punter in the city centre. If that sounds okay, what day and time would you like?
            USER	Yes please! Thank you so much!
            SYSTEM	I am sorry but you will have to get ticket there and here is the address 251a chesterton road.
            USER	I am also looking for the Oak Bistro, a place to eat.

            ### Step 1: Extract all of the main goals of the user from the dialogue.
            1. **Find and make reservation for an attraction**
            2. **Find information about "Oak Bistro," a place to eat.**

            ### Step 2: For each goal, Classify the efficiency of system handling the goal.
            1. **Goal: Find and get information about a boat attraction in the city centre.**
            - Efficiency: Inefficient
            - Reason: The system offers to get tickets but then redirects the user to get tickets on-site without fulfilling the initial request

            2. **Goal: Find information about "Oak Bistro," a place to eat.**
            - Efficiency: Not evaluated
            - Reason: The dialogue ends abruptly.

            ### Step 3: For each goal, classify it as positive, negative, or neutral.
            1. **Find and get information about a boat attraction in the city centre:**
            - Classification: Negative
            - Reason: The system redirects the user to get tickets on-site without fulfilling the initial goal efficiently.

            2. **Find information about "Oak Bistro," a place to eat:**
            - Classification: Negative
            - Reason: The system does not respond to this request at all, leaving the goal unfulfilled.

            ### Step 4: User Satisfaction: 1 
            - Reason: Both goals are classified as negative. The system failed to fulfill the user's requests efficiently or completely. The first goal was mishandled, and the second was ignored entirely.


            Example 3:
            USER	I need help finding a hotel that has free wifi and parking, can you help?
            SYSTEM	The Ashley Hotel and Lovell Lodge are moderately priced hotels, and the Express by Holiday Cambridge, Gonville Hotel, and Huntingdon Marriott are more expensively priced. All have parking and wifi.
            USER	Actually, are any of these 4 star rated?
            SYSTEM	The Huntingdon Marriott Hotel is a four star hotel.
            USER	Does the hotel provide breakfast buffet?
            SYSTEM	Yes, the Huntingdon Marriott Hotel serves buffet breakfast. How does that sound?	
            USER	Sounds perfect. Can you book a room for 2 nights starting Sunday? There are 7 of us.
            SYSTEM	Yes I have done so and your reference number is U2UTF524.
            USER	I'm also looking for a place to dine that's in the centre and in the expensive range.
            SYSTEM	The bedouin features african cusine and is located in the centre in the expensive range.
            USER	Is there anything that serves Turkish food?
            SYSTEM	Yes there is the meze bar restaurant would you like me to book it for you?	
            USER	No, I just need the phone number.
            SYSTEM	I'm sorry, but Meze Bar Restaurant's phone number is not listed.	
            USER	I need a taxi from Huntingdon Marriott Hotel to Meze Bar Restaurant
            SYSTEM	What time do you want to leave the hotel?
            USER	Sounds good if I can stay with my kid.
            SYSTEM	The Huntingdon Marriot welcomes children to stay with them. Do you want me to book you a room?		
            USER	Doesn't matter. I just need to leave the restaurant by 13:00. I will also need the contact number and car type of the taxi.
            SYSTEM	Sure thing. A yellow Honda will be outside waiting for you. The driver's contact number is 07929295523. Can I help with anything else?
            USER	If I need to change my taxi booking, will I be able to do that?
            SYSTEM	Yes, you may cancel up to 24 hours for free, only minor changes are allowed after this time frame. Anything else I can do for you?
            USER	That will be all. Thanks for your help!

            ### Step 1: Extract all of the main goals of the user from the dialogue.
            1. Find and book a hotel
            2. Find a Turkish restaurant
            3. Arrange a taxi

            ### Step 2: For each goal, Classify the efficiency of system handling the goal.
            1. **Goal: Find and book a hotel**
            - Efficiency: Efficient
            - Reason: The system quickly provided options, confirmed the 4-star rating and breakfast buffet, and successfully booked the room without unnecessary turns.

            2. **Goal: Find a Turkish restaurant.**
            - Efficiency: Efficient
            - Reason: The system promptly suggested a Turkish restaurant (Meze Bar Restaurant) that met the criteria, though the phone number was not available.

            3. **Goal: Arrange a taxi.**
            - Efficiency: Efficient
            - Reason: The system efficiently arranged the taxi, provided the contact number and car type, and clarified the booking change policy.

            ### Step 3: For each goal, classify it as positive, negative, or neutral.
            1. **Find and book a hotel.**
            - Classification: Positive
            - Reason: The system fulfilled all the user's requirements efficiently.

            2. **Find a Turkish restaurant in the center and in the expensive range.**
            - Classification: Neutral
            - Reason: The system found a suitable restaurant but could not provide the phone number.

            3. **Arrange a taxi from the hotel to the restaurant, including contact number and car type.**
            - Classification: Positive
            - Reason: The system efficiently arranged the taxi and provided all requested details.

            ### Step 4: User Satisfaction: 2
            Reason: Two out of Three goals were classified as positive, and one was neutral. 


            Example 4:
            USER	Hi there. Can you help me find a train to Cambridge on Thursday?
            SYSTEM	I can. Will you be departing from London Kings Cross?
            USER	No, I'm coming from Ely.
            SYSTEM	What time would you like to travel?	
            USER	I would like to arrive in Cambridge by 16:15.
            SYSTEM	I would recommend the TR7745 train that arrives at 07:52. Would that work?	
            USER	I need the travel time for that.
            SYSTEM	It is 17 minutes.	
            USER	Okay, if you book it can I change my ticket later if needed?
            SYSTEM	No, you would need to get the ticket refunded and then buy a new one. Is there anything else I can help with?		
            USER	Does the train have WiFi service that I can use while riding?
            SYSTEM	Yes there will be wifi on the train. Log on with ID GreatNorthern_WiFi. Would you like any other info?		
            USER	can you see about reservations at the aylesbray lodge guest house
            SYSTEM	Yes, I can. Will it be for for just 1 night?	
            USER	Do you have to pay to park in their parking lot there?
            SYSTEM	The parking in this establishment is free. How else can I help you today?		
            USER	Actually, let's not worry about booking it just now. Can you narrow down where it is for me with the area at postcode?
            SYSTEM	Sure. It's in the south of town with postcode cb17sr. Can I help you with anything else?
            USER	No, that seems like everything. Go ahead and book it, and we should be done here.
            SYSTEM	How many nights will you be staying?
            USER	I'm sorry, I'm not ready to book yet. I can do it myself later. Thanks so much for all of your help. Bye.


            ### Step 1: Extract all of the main goals of the user from the dialogue.
            1. **Find and book a train.**
            2. **Get information about the Aylesbray Lodge Guest House (including parking and location).**

            ### Step 2: For each goal, Classify the efficiency of system handling the goal.
            1. **Goal: Find and book a train from Ely to Cambridge.**
            - Efficiency: Efficient
            - Reason: The system quickly identifies the user's departure point (Ely) and desired arrival time, provides a suitable train option, and answers follow-up questions about WiFi and ticket changes without unnecessary turns.

            2. **Goal: Get information about the Aylesbray Lodge Guest House.**
            - Efficiency: Efficient
            - Reason: The system promptly addresses the user's queries about parking and location (postcode) and clarifies booking details when the user hesitates. No unnecessary turns or misunderstandings occur.

            ### Step 3: For each goal, classify it as positive, negative, or neutral.
            1. **Find and book a train from Ely to Cambridge:**
            - Classification: Positive
            - Reason: The system fulfills the main goal (providing train options and answering related questions) efficiently. The user does not proceed with booking, but this is due to the user's decision, not system failure.

            2. **Get information about the Aylesbray Lodge Guest House:**
            - Classification: Positive
            - Reason: The system provides all requested information (parking, location/postcode) efficiently. The user decides not to book, but this is unrelated to the system's performance.

            ### Step 4: User Satisfaction: 3
            - Reason: Both goals are handled efficiently and fulfilled to the user's satisfaction. The user's decision not to book does not reflect negatively on the system's performance.


            Example 5:
            USER	I am looking for a entertainment in the south area.
            SYSTEM	Nusha is an entertainment attraction in the South. Would you like more information?	
            USER	Yes, what's their postcode? And while you're at it, maybe their phone number, too.
            SYSTEM	The postcode is cb17dy and the phone number is 01223902158. Is there anything else I can help you with?	
            USER	I was also wondering if you can help me find a restaurant called tandoori palace.
            SYSTEM	Yes. It's an Indian restaurant in the west part of town at 68 Histon Road Chesterton. Would you like to book a table?
            USER	Does this restaurant allow customers to BYOB?
            SYSTEM	Unfortunately you can't bring your own bottles. Do you have any additional questions?		
            USER	Does this restaurant have free WiFi for customers?
            SYSTEM	I'm sorry they don't have any WiFi. Was there anything else you needed?		
            USER	Does this restaurant offer outdoor seating?
            SYSTEM	I'm sorry to say that Outdoor seating is not available at Tandoori Palace. We have a wonderful indoor setting if you'd like to see.	
            USER	Can you give me the postal code for tandoori palace?
            SYSTEM	The post code for tandoori palace is cb43le. Can I book a table, if so for how many on what date?	
            USER	No need to book a table at this time. Thank you for your help. Good bye.


            ### Step 1: Extract all of the main goals of the user from the dialogue.
            1. **Find information about an entertainment attraction (Nusha).**
            2. **Find information about a restaurant called Tandoori Palace.**

            ### Step 2: For each goal, Classify the efficiency of system handling the goal.
            1. **Goal: Find information about Nusha.**
            - Efficiency: Efficient
            - Reason: The system provided all requested information efficiently

            2. **Goal: Find information about Tandoori Palace.**
            - Efficiency: Efficient
            - Reason: The system provided all requested information efficiently

            ### Step 3: For each goal, classify it as positive, negative, or neutral.
            1. **Find information about Nusha:**
            - Classification: Positive
            - Reason: The system fulfilled goals efficiently.

            2. **Find information about Tandoori Palace:**
            - Classification: Positive
            - Reason: The system fulfilled all goals.

            ### Step 4: User Satisfaction: 3
            Reason: All goals were classified as positive.
            """
        import http.client
        import json

        #model_choice = 'ark-deepseek-r1-250120'
        #model_choice = 'gpt-4o'
        #model_choice = 'mistral-large-latest'
        #model_choice = 'claude-3-7-sonnet-latest'
        api_key = ""
        if openAIport:
            print("Using OPENAI")
            client = OpenAI(
                api_key=""
            )
            response1 = client.chat.completions.create(
            model="gpt-4.1",
            #model = "gpt-4o",
              messages=[
                {"role": "system", "content": system_prompt
                 },
                {"role": "user", "content": "Given the dialogue: \n" + dialogue +"\nPlease evaluate the user satisfaction rate according to the format."}
            ]
            )

            # Content of response
            content1 = response1.choices[0].message.content

            with open(output_dir + str(i) + ".txt", "w+") as f:
                f.write(content1)
                # f.write("\nStep 3: \n" + content2)
        else:
            conn = http.client.HTTPSConnection("c-z0-api-01.hash070.com")
            payload = json.dumps({
            "model": model_choice,
            'temperature': 0.2,
            'top_p': 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": "Given the dialogue: \n" + dialogue +"\nPlease evaluate the user satisfaction rate according to the format."
                }
            ],
            "stream": False
            })

            headers = {
            'Content-Type': 'application/json',
            "Authorization": 'Bearer ' + api_key,
            }

        try:
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            data = res.read()
            data_json = json.loads(data.decode("utf-8")) 
            print(data.decode("utf-8"))
            
    
            # Content of response
            #content1 = response1.choices[0].message.content
            content1 = data_json['choices'][0]['message']['content']
            with open(output_dir + str(i) + ".txt", "w+") as f:
                f.write(content1)
                #f.write("\nStep 3: \n" + content2)
            idx_except = 0
        except:
            idx_except += 1
            if idx_except >= 5:
                fail_idx.append(i)
                pass
            else:
                i = i - 1
            print("Exception Occur:", idx_except)
        

 

if __name__ == '__main__':
    main()