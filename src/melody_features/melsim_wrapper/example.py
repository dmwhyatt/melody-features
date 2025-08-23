from melody_features.melsim_wrapper.melsim import (
    get_similarity_from_midi,
    install_dependencies,
)

if __name__ == "__main__":
    # Install dependencies
    install_dependencies()

    # example files from Essen folksong corpus
    appenzel_path = "Feature_Set/melsim_wrapper/appenzel.mid"
    arabic_path = "Feature_Set/melsim_wrapper/arabic01.mid"

    # if you wished to use a directory, you would supply it like so:
    midi_dir = "PATH_TO_MIDI_DIRECTORY"

    # Calculate similarity between two MIDI files
    similarity_value = get_similarity_from_midi(
        appenzel_path,
        arabic_path,
        method="Jaccard",  # Using Jaccard similarity measure
        transformation="pitch",  # Compare raw pitch values
    )
    print(f"Jaccard pitch similarity: {similarity_value:.3f}")

    # Try another combination
    similarity_value = get_similarity_from_midi(
        appenzel_path,
        arabic_path,
        method="edit_sim",  # Using edit distance similarity
        transformation="parsons",  # Compare melodic contours
    )
    print(f"Edit distance similarity using Parsons code: {similarity_value:.3f}")

    # example of using a directory and multiple methods and transformations

    midi_corpus_similarity = get_similarity_from_midi(
        midi_dir,
        transformation=["pitch", "parsons"],
        method=["Jaccard", "edit_sim"],
        output_file="midi_corpus_similarity.json",
    )
