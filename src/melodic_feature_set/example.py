from melodic_feature_set.features import get_all_features, Config, IDyOMConfig, FantasticConfig

# Example usage of the Config class
config = Config(
    # Setting this to None will skip corpus-dependent features, unless
    # we supply a corpus path in the idyom or fantastic configs.
    corpus="src/melodic_feature_set/Essen_Corpus",
    # We can supply multiple IDyOM configs using a dictionary
    # this means we can use different corpora and viewpoints for each config
    idyom={"pitch": IDyOMConfig(
        target_viewpoints=["cpitch"],
        source_viewpoints=["cpint", "cpintfref"],
        ppm_order=2,
        corpus="src/melodic_feature_set/Essen_Corpus",
        models=":both"
    ),
    "rhythm": IDyOMConfig(
        target_viewpoints=["onset"],
        source_viewpoints=["ioi"],
        ppm_order=1,
        corpus="src/melodic_feature_set/Essen_Corpus",
        models=":both"
    )},
    # Omitting the corpus path in Fantastic here will
    # use the corpus path from the Config object instead.
    fantastic=FantasticConfig(
        max_ngram_order=6,
        phrase_gap=1.5
    )
)

get_all_features(input_directory="PATH",
                output_file="example.csv",
                config=config)
