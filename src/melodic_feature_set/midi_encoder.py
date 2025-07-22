import csv
import json
import pretty_midi



class OriginalMelody:

    def __init__(self, file):
        self.file = file
        self.midi_data = pretty_midi.PrettyMIDI(file).instruments[0].notes

    def transpose(self, amount):
        for note in self.midi_data:
            note.pitch += amount
        return self.midi_data


class ModifiedMelody(OriginalMelody):

    def __init__(self, in_key, contour_dif, change_note, displacement, oddity, discrimination, difficulty):
        super().__init__(file)
        self.in_key = in_key
        self.contour_dif = contour_dif
        # change note is 1-indexed
        self.change_note = change_note - 1
        self.displacement = displacement
        self.oddity = oddity
        self.discrimination = discrimination
        self.difficulty = difficulty

    def directionFinder(self):
        contour_scalar = self.midi_data[(self.change_note - 2)].pitch - self.midi_data[(self.change_note - 1)].pitch
        return contour_scalar

    def changeMelody(self, amount):
        if self.directionFinder() > 0 & contour_dif == 0:
            self.midi_data[self.change_note].pitch -= displacement

        elif self.directionFinder() > 0 & contour_dif == 4:
            self.midi_data[self.change_note].pitch += displacement

        elif self.directionFinder() <= 0 & contour_dif == 0:
            self.midi_data[self.change_note].pitch += displacement

        elif self.directionFinder() <= 0 & contour_dif == 4:
            self.midi_data[self.change_note].pitch -= displacement

        self.transpose(amount)
        return self.midi_data


def create_sequence(file, in_key, contour_dif, change_note, displacement, oddity, discrimination,
                    difficulty):
    if oddity == 1:
        first = ModifiedMelody(in_key, contour_dif, change_note, displacement, oddity, discrimination, difficulty)
        second = OriginalMelody(file)
        third = OriginalMelody(file)

        first.midi_data.pop()
        second.midi_data.pop()
        third.midi_data.pop()

        stimuli = first.changeMelody(0) + second.transpose(1) + third.transpose(2)
        return stimuli

    elif oddity == 2:
        first = OriginalMelody(file)
        second = ModifiedMelody(in_key, contour_dif, change_note, displacement, oddity, discrimination, difficulty)
        third = OriginalMelody(file)

        first.midi_data.pop()
        second.midi_data.pop()
        third.midi_data.pop()

        stimuli = first.midi_data + second.changeMelody(1) + third.transpose(2)
        return stimuli

    elif oddity == 3:
        first = OriginalMelody(file)
        second = OriginalMelody(file)
        third = ModifiedMelody(in_key, contour_dif, change_note, displacement, oddity, discrimination, difficulty)

        first.midi_data.pop()
        second.midi_data.pop()
        third.midi_data.pop()

        stimuli = first.midi_data + second.transpose(1) + third.changeMelody(2)
        return stimuli


with open("item-bank.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    output_file = open("mididata5.json", "a", encoding='utf-8')
    output_file.write("[\n")
    count = 1

    for row in reader:
        name = str(row["original_melody"])
        in_key = str(row["in_key"])
        contour_dif = int(row["contour_dif"])
        change_note = int(row["change_note"])
        displacement = int(row["displacement"])
        oddity = int(row["oddity"])
        discrimination = float(row["discrimination"])
        difficulty = float(row["difficulty"])

        file = "mid/{melody}.mid".format(melody=name)
        # print(create_sequence(file, in_key, contour_dif, change_note, displacement, oddity, discrimination,
        #                      difficulty))

        output = str(create_sequence(file, in_key, contour_dif, change_note, displacement, oddity, discrimination,
                                     difficulty))[1:-1]

        output_string = {"ID": count,
                         "Original Melody": name,
                         "In Key": in_key,
                         "Contour Dif": contour_dif,
                         "Change Note": change_note,
                         "Displacement": displacement,
                         "Oddity": oddity,
                         "Discrimination": discrimination,
                         "Difficulty": difficulty,
                         "MIDI Sequence": output,
                         }
        output_dict = json.dumps(output_string, indent=4)
        output_file.write(output_dict)
        output_file.write(",\n")
        count += 1

        # json.dump(output_string, output_file, indent=4)

    output_file.write("]")
    output_file.close()
