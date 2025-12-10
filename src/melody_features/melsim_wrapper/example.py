from melody_features.melsim_wrapper.melsim import (
    get_similarity_from_midi,
    install_dependencies,
)
from melody_features.corpus import get_corpus_files

if __name__ == "__main__":
    # Install dependencies
    install_dependencies()

    # example files from Essen folksong corpus
    appenzel_path = "src/melody_features/corpora/essen_folksong_collection/appenzel.mid"
    arabic_path = "src/melody_features/corpora/essen_folksong_collection/arabic01.mid"

    # if you wished to, you could instead use a directory:
    midi_dir = "src/melody_features/corpora/essen_folksong_collection"

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
    # Get first 10 files from the Essen corpus
    first_10_files = get_corpus_files("essen", max_files=10)
    print(f"Using first 10 files from Essen corpus: {[f.name for f in first_10_files]}")

    midi_corpus_similarity = get_similarity_from_midi(
        first_10_files,  # Use the list of first 10 files
        midi_path2=None,  # Not needed for directory processing
        transformation=["pitch", "parsons"],
        method=["Jaccard", "edit_sim"],
        output_file="midi_corpus_similarity.json",
    )
