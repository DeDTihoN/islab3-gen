import copy
import random
import pandas as pd

# Генерація предметів з кількістю занять
def generate_subjects(P, T):
    subjects = []
    for i in range(P):
        total_classes = random.randint(T-2, T)
        lecture_hours = random.randint(1, total_classes)
        practical_hours = total_classes - lecture_hours
        subjects.append({
            "subject": f"Subject_{i + 1}",
            "lecture_hours": lecture_hours,
            "practical_hours": practical_hours
        })
    return subjects

# Генерація початкових даних
def generate_data(G, N, P, T, L, A):
    # Генеруємо список предметів
    subjects = generate_subjects(P, T)

    # Генеруємо список груп із кількістю студентів і предметами
    groups = []
    for i in range(G):
        subject_subset = random.sample(subjects, k=random.randint(P-2,P))
        groups.append({"group_id": i + 1, "students": N, "subjects": subject_subset})

    # Генеруємо список лекторів із підмножинами предметів, які вони можуть викладати
    lecturers = []
    for i in range(L):
        subject_subset = random.sample(subjects, k=random.randint(1, len(subjects)))
        lecturer_subjects = {}
        for subject in subject_subset:
            lecturer_subjects[subject["subject"]] = {
                "can_lecture": random.choice([True, False]),
                "can_practice": random.choice([True, False])
            }
        lecturers.append({
            "lecturer_id": i + 1,
            "subjects": lecturer_subjects
        })

    # Генеруємо список аудиторій з місткістю
    auditoriums = [{"auditorium_id": i + 1, "capacity": random.randint(N, N * 4)} for i in range(A)]

    return groups, subjects, lecturers, auditoriums


# Функція для генерації розкладу
def generate_schedule(groups, subjects, lecturers, auditoriums):
    # Створюємо порожній розклад
    schedule = {slot: [] for slot in range(1, 21)}  # 20 слотів на тиждень

    # Проходимо по кожному слоту
    for slot in range(1, 21):
        for group in groups:
            # Імовірність, що у групи не буде заняття в цьому слоті
            if random.random() < no_class_probability:  # 10% ймовірність пропуску
                continue

            # Вибираємо доступних лекторів, які ще не зайняті у цьому слоті практичними заняттями
            available_lecturers = [
                lecturer for lecturer in lecturers
                if not any(
                    entry["lecturer_id"] == lecturer["lecturer_id"] and entry["type"] == "Practical"
                    for entry in schedule[slot]
                )
            ]

            if not available_lecturers:
                continue  # Якщо немає доступних лекторів, пропускаємо цей слот для групи

            # Вибираємо випадкового викладача з доступних
            lecturer = random.choice(available_lecturers)
            group_schedule = None

            # Перевіряємо, чи лектор вже веде заняття в цей день
            for entry in schedule[slot]:
                if entry["lecturer_id"] == lecturer["lecturer_id"]:
                    group_schedule = entry
                    break

            if group_schedule:  # Якщо лектор вже веде заняття
                # Додаємо групу до існуючого розкладу
                schedule[slot].append({
                    "group_id": group["group_id"],
                    "subject": group_schedule["subject"],
                    "lecturer_id": lecturer["lecturer_id"],
                    "type": group_schedule["type"],
                    "auditorium_id": group_schedule["auditorium_id"]
                })
            else:  # Інакше вибираємо новий предмет і тип заняття
                # Вибираємо випадковий предмет, який лектор може викладати
                available_subjects = [
                    s for s in lecturer["subjects"].items() if s[1]["can_lecture"] or s[1]["can_practice"]
                ]
                if not available_subjects:
                    continue

                subject, abilities = random.choice(available_subjects)
                type_of_class = "Lecture" if abilities["can_lecture"] and random.choice([True, False]) else "Practical"

                # Вибираємо незайняту аудиторію
                occupied_auditoriums = {entry["auditorium_id"] for entry in schedule[slot]}
                available_auditoriums = [aud for aud in auditoriums if aud["auditorium_id"] not in occupied_auditoriums]

                if not available_auditoriums:
                    continue  # Якщо немає доступних аудиторій, пропускаємо

                auditorium = random.choice(available_auditoriums)

                # Додаємо новий запис у розклад
                schedule[slot].append({
                    "group_id": group["group_id"],
                    "subject": subject,
                    "lecturer_id": lecturer["lecturer_id"],
                    "type": type_of_class,
                    "auditorium_id": auditorium["auditorium_id"]
                })

    return schedule

def calculate_fitness(schedule, groups, lecturers, auditoriums):
    fitness = 0

    # Створюємо словники для зберігання кількості занять для кожної групи
    group_lecture_count = {group["group_id"]: {} for group in groups}
    group_practical_count = {group["group_id"]: {} for group in groups}

    # Ініціалізуємо структуру для відстеження розривів у розкладі груп і викладачів
    group_slots = {group["group_id"]: [] for group in groups}
    lecturer_slots = {lecturer["lecturer_id"]: [] for lecturer in lecturers}  # Виправлення тут

    # Заповнюємо структури, щоб обчислити кількість занять і розриви
    for slot, classes in schedule.items():
        for entry in classes:
            group_id = entry["group_id"]
            lecturer_id = entry["lecturer_id"]
            subject = entry["subject"]
            type_of_class = entry["type"]
            auditorium_id = entry["auditorium_id"]

            # Оновлюємо кількість занять для групи
            if type_of_class == "Lecture":
                group_lecture_count[group_id][subject] = group_lecture_count[group_id].get(subject, 0) + 1
            else:
                group_practical_count[group_id][subject] = group_practical_count[group_id].get(subject, 0) + 1

            # Оновлюємо слоти для розрахунку розривів
            group_slots[group_id].append(slot)
            if lecturer_id not in lecturer_slots:
                lecturer_slots[lecturer_id] = []
            lecturer_slots[lecturer_id].append(slot)

            # Перевіряємо заповнюваність аудиторії
            auditorium = next((aud for aud in auditoriums if aud["auditorium_id"] == auditorium_id), None)
            if auditorium:
                if auditorium["capacity"] < groups[group_id - 1]["students"]:
                    # Додаємо штраф за переповнення аудиторії
                    fitness += (groups[group_id - 1]["students"] - auditorium["capacity"]) / groups[group_id - 1]["students"]

    # Розрахунок штрафів за невідповідність кількості лекцій і практичних занять
    for group in groups:
        group_id = group["group_id"]
        for subject in group["subjects"]:
            required_lecture_hours = subject["lecture_hours"]
            required_practical_hours = subject["practical_hours"]

            actual_lecture_hours = group_lecture_count[group_id].get(subject["subject"], 0)
            actual_practical_hours = group_practical_count[group_id].get(subject["subject"], 0)

            # Додаємо абсолютну різницю між необхідною та фактичною кількістю занять
            fitness += abs(required_lecture_hours - actual_lecture_hours)
            fitness += abs(required_practical_hours - actual_practical_hours)

    # Підрахунок штрафів для груп
    for slots in group_slots.values():
        fitness += calculate_day_gaps(slots)

    # Підрахунок штрафів для викладачів
    for slots in lecturer_slots.values():
        fitness += calculate_day_gaps(slots)

    return fitness


# Розрахунок штрафів за розриви в розкладі для груп і викладачів
def calculate_day_gaps(slots):
    # Сортуємо слоти
    slots.sort()

    # Розбиваємо слоти на блоки по 4, відповідно до множин (1-4, 5-8, 9-12, 13-16, 17-20)
    gaps = 0
    daily_blocks = [
        set(range(1, 5)),  # Блок для слотів 1-4
        set(range(5, 9)),  # Блок для слотів 5-8
        set(range(9, 13)),  # Блок для слотів 9-12
        set(range(13, 17)),  # Блок для слотів 13-16
        set(range(17, 21))  # Блок для слотів 17-20
    ]

    for block in daily_blocks:
        # Фільтруємо слоти, які потрапляють у поточний блок
        block_slots = [slot for slot in slots if slot in block]

        # Якщо є більше одного заняття в блоці, перевіряємо розриви
        if len(block_slots) > 1:
            for i in range(1, len(block_slots)):
                if block_slots[i] - block_slots[i - 1] > 1:  # Перевірка на наявність розриву
                    gaps += 1
    return gaps


# Функція для виконання кросоверу між двома розкладами
def crossover(schedule1, schedule2):
    # Створюємо копії розкладів, щоб не змінювати оригінали
    child_schedule = copy.deepcopy(schedule1)

    # # Визначаємо точку кросоверу
    # crossover_point = random.randint(1, 19)  # Вибираємо точку кросоверу (1-19)
    #
    # # Міняємо слоти між розкладами
    # for slot in range(crossover_point, 21):
    #     child_schedule[slot] = copy.deepcopy(schedule2[slot])

    for slot in range(1, 21):
        if random.choice([True, False]):
            child_schedule[slot] = copy.deepcopy(schedule2[slot])

    return child_schedule

def mutate(schedule, groups, subjects, lecturers, auditoriums):
    # Вибираємо випадковий слот для мутації
    slot_to_mutate = random.randint(1, 20)
    slot = slot_to_mutate
    if random.random() < mut_prob:  # 10% ймовірність пропуску
        schedule[slot] = []
        for group in groups:
            # Імовірність, що у групи не буде заняття в цьому слоті
            if random.random() < no_class_probability:  # 10% ймовірність пропуску
                continue

            # Вибираємо доступних лекторів, які ще не зайняті у цьому слоті практичними заняттями
            available_lecturers = [
                lecturer for lecturer in lecturers
                if not any(
                    entry["lecturer_id"] == lecturer["lecturer_id"] and entry["type"] == "Practical"
                    for entry in schedule[slot]
                )
            ]

            if not available_lecturers:
                continue  # Якщо немає доступних лекторів, пропускаємо цей слот для групи

            # Вибираємо випадкового викладача з доступних
            lecturer = random.choice(available_lecturers)
            group_schedule = None

            # Перевіряємо, чи лектор вже веде заняття в цей день
            for entry in schedule[slot]:
                if entry["lecturer_id"] == lecturer["lecturer_id"]:
                    group_schedule = entry
                    break

            if group_schedule:  # Якщо лектор вже веде заняття
                # Додаємо групу до існуючого розкладу
                schedule[slot].append({
                    "group_id": group["group_id"],
                    "subject": group_schedule["subject"],
                    "lecturer_id": lecturer["lecturer_id"],
                    "type": group_schedule["type"],
                    "auditorium_id": group_schedule["auditorium_id"]
                })
            else:  # Інакше вибираємо новий предмет і тип заняття
                # Вибираємо випадковий предмет, який лектор може викладати
                available_subjects = [
                    s for s in lecturer["subjects"].items() if s[1]["can_lecture"] or s[1]["can_practice"]
                ]
                if not available_subjects:
                    continue

                subject, abilities = random.choice(available_subjects)
                type_of_class = "Lecture" if abilities["can_lecture"] and random.choice([True, False]) else "Practical"

                # Вибираємо незайняту аудиторію
                occupied_auditoriums = {entry["auditorium_id"] for entry in schedule[slot]}
                available_auditoriums = [aud for aud in auditoriums if aud["auditorium_id"] not in occupied_auditoriums]

                if not available_auditoriums:
                    continue  # Якщо немає доступних аудиторій, пропускаємо

                auditorium = random.choice(available_auditoriums)

                # Додаємо новий запис у розклад
                schedule[slot].append({
                    "group_id": group["group_id"],
                    "subject": subject,
                    "lecturer_id": lecturer["lecturer_id"],
                    "type": type_of_class,
                    "auditorium_id": auditorium["auditorium_id"]
                })
    return schedule

# Функція для генерації початкової популяції
def generate_initial_population(size, groups, subjects, lecturers, auditoriums):
    population = []
    for _ in range(size):
        schedule = generate_schedule(groups, subjects, lecturers, auditoriums)
        population.append(schedule)
    return population

# Основна функція генетичного алгоритму
def genetic_algorithm(groups, subjects, lecturers, auditoriums, generations=100, population_size=300):
    # Генеруємо початкову популяцію
    population = generate_initial_population(population_size, groups, subjects, lecturers, auditoriums)

    best_schedule_of_all_generations = None
    best_fitness_of_all_generations = 1e9
    # Основний цикл генетичного алгоритму
    for generation in range(generations):
        # Оцінюємо придатність кожного розкладу
        fitness_scores = [calculate_fitness(schedule, groups, lecturers, auditoriums) for schedule in population]

        # Відбираємо найкращі розклади
        sorted_population = [schedule for _, schedule in sorted(zip(fitness_scores, population), key=lambda x: x[0])]
        population = sorted_population[:population_size // 2]  # Вибираємо половину кращих

        # Виконуємо кросовер і мутацію, щоб створити нові розклади
        new_population = []
        while len(new_population) < population_size:
            # Вибираємо двох випадкових батьків
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            # Виконуємо кросовер
            child = crossover(parent1, parent2)
            # Виконуємо мутацію
            child = mutate(child, groups, subjects, lecturers, auditoriums)
            new_population.append(child)

        # Оновлюємо популяцію
        population = new_population

        # Виводимо інформацію про найкращий розклад у поточному поколінні

        best_fitness = min(fitness_scores)
        if best_fitness < best_fitness_of_all_generations:
            best_fitness_of_all_generations = best_fitness
            best_schedule_of_all_generations = sorted_population[0]
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
        if best_fitness == 0:
            break
    # Повертаємо найкращий розклад
    return best_schedule_of_all_generations

# Функція для збереження даних у CSV
def save_data_to_csv(groups, subjects, lecturers, auditoriums, schedule):
    # Збереження предметів
    subjects_df = pd.DataFrame(subjects)
    subjects_df.to_csv("subjects.csv", index=False)

    # Збереження груп
    groups_data = []
    for group in groups:
        for subject in group["subjects"]:
            groups_data.append({
                "group_id": group["group_id"],
                "students": group["students"],
                "subject": subject["subject"],
                "lecture_hours": subject["lecture_hours"],
                "practical_hours": subject["practical_hours"]
            })
    groups_df = pd.DataFrame(groups_data)
    groups_df.to_csv("groups.csv", index=False)

    # Збереження лекторів
    lecturers_data = []
    for lecturer in lecturers:
        for subject, abilities in lecturer["subjects"].items():
            lecturers_data.append({
                "lecturer_id": lecturer["lecturer_id"],
                "subject": subject,
                "can_lecture": abilities["can_lecture"],
                "can_practice": abilities["can_practice"]
            })
    lecturers_df = pd.DataFrame(lecturers_data)
    lecturers_df.to_csv("lecturers.csv", index=False)

    # Збереження аудиторій
    auditoriums_df = pd.DataFrame(auditoriums)
    auditoriums_df.to_csv("auditoriums.csv", index=False)

    # Збереження розкладу
    schedule_data = []
    for slot, classes in schedule.items():
        for entry in classes:
            schedule_data.append({
                "slot": slot,
                "group_id": entry["group_id"],
                "subject": entry["subject"],
                "lecturer_id": entry["lecturer_id"],
                "type": entry["type"],
                "auditorium_id": entry["auditorium_id"]
            })
    schedule_df = pd.DataFrame(schedule_data)
    schedule_df.to_csv("schedule.csv", index=False)

# Функція для форматування розкладу по днях тижня
def format_schedule_for_excel(schedule, groups):
    # Створюємо словник для збереження розкладу по групах
    formatted_schedules = {}

    # Проходимо по кожній групі
    for group in groups:
        group_id = group["group_id"]
        # Створюємо словник для розкладу по днях тижня
        weekly_schedule = {
            "Понеділок": {i: "" for i in range(1, 5)},
            "Вівторок": {i: "" for i in range(5, 9)},
            "Середа": {i: "" for i in range(9, 13)},
            "Четвер": {i: "" for i in range(13, 17)},
            "П’ятниця": {i: "" for i in range(17, 21)}
        }

        # Заповнюємо розклад для поточної групи
        for slot, classes in schedule.items():
            for entry in classes:
                if entry["group_id"] == group_id:
                    ind = (slot-1) % 4 + 1
                    if 1 <= slot <= 4:
                        weekly_schedule["Понеділок"][ind] = f"{entry['subject']} ({entry['type']}) - Лектор {entry['lecturer_id']} - Аудиторія {entry['auditorium_id']}"
                    elif 5 <= slot <= 8:
                        weekly_schedule["Вівторок"][ind] = f"{entry['subject']} ({entry['type']}) - Лектор {entry['lecturer_id']} - Аудиторія {entry['auditorium_id']}"
                    elif 9 <= slot <= 12:
                        weekly_schedule["Середа"][ind] = f"{entry['subject']} ({entry['type']}) - Лектор {entry['lecturer_id']} - Аудиторія {entry['auditorium_id']}"
                    elif 13 <= slot <= 16:
                        weekly_schedule["Четвер"][ind] = f"{entry['subject']} ({entry['type']}) - Лектор {entry['lecturer_id']} - Аудиторія {entry['auditorium_id']}"
                    elif 17 <= slot <= 20:
                        weekly_schedule["П’ятниця"][ind] = f"{entry['subject']} ({entry['type']}) - Лектор {entry['lecturer_id']} - Аудиторія {entry['auditorium_id']}"

        # Перетворюємо розклад у DataFrame
        df = pd.DataFrame(weekly_schedule)
        formatted_schedules[f"Group_{group_id}"] = df

    return formatted_schedules

# Функція для збереження розкладу у багатотабличний Excel файл
def save_schedule_to_excel(schedule, groups):
    formatted_schedules = format_schedule_for_excel(schedule, groups)

    # Зберігаємо всі розклади в одному Excel файлі
    with pd.ExcelWriter("group_schedules.xlsx") as writer:
        for group_name, df in formatted_schedules.items():
            df.to_excel(writer, sheet_name=group_name, index=False)

# Генерація початкових даних
G = 3  # кількість груп
N = 30  # кількість студентів у групі
P = 5  # загальна кількість предметів
T = 4  # максимальна кількість занять на предмет
L = 10  # кількість лекторів
A = 7   # кількість аудиторій
mut_prob = 0.2
no_class_probability = 0.15


groups, subjects, lecturers, auditoriums = generate_data(G, N, P, T, L, A)

# Вивід даних для перевірки
print("Subjects:", subjects)
print("Groups:", groups)
print("Lecturers:", lecturers)
print("Auditoriums:", auditoriums)


# Викликаємо генетичний алгоритм
best_schedule = genetic_algorithm(groups, subjects, lecturers, auditoriums)

# Виводимо найкращий розклад
print("Best Schedule:", best_schedule)

# Викликаємо функцію для збереження даних у CSV
save_data_to_csv(groups, subjects, lecturers, auditoriums, best_schedule)

print("Дані успішно збережені в CSV файли!")


# Викликаємо функцію для збереження розкладу
save_schedule_to_excel(best_schedule, groups)

print("Розклад успішно збережено у файлі group_schedules.xlsx!")