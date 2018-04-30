import binarization
import feature_extraction
import feature_norm
import DTWDistance


def main():
    binarization.binarize()
    words = binarization.cropImages()
    features = feature_extraction.CreateAllFeatures(words)
    features_normed_linear = feature_norm.fNormLinear(features)
    features_normed_non_linear = feature_norm.fNormNonLinear(features)
    dtw_distance = DTWDistance.DTWDistance(features_normed_linear, features_normed_non_linear)
    print(dtw_distance)

if __name__ == "__main__":
    main()