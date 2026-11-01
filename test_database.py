from database import create_table, save_detection, get_history

create_table()

save_detection(
    "SMS",
    "Congratulations! You won a prize!",
    "Spam",
    98.5
)

print(get_history())