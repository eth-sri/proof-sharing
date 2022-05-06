#include "gurobi_c++.h"
#include "abstraction/abstraction.h"
#include "domains/polyhedra.h"
#include "utils/lipschitz.h"
#include "utils/constants.h"
#include "transforms/transformation.h"
#include "transforms/interpolation.h"
#include "transforms/parser.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <map>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <sstream>

using namespace std;

const GRBEnv& env = GRBEnv("gurobi.log");

vector<pair<PointD, Image>> generateAttacks(
        ofstream& out,
        const HyperBox& combinedDomain,
        const SpatialTransformation& spatialTransformation,
        const PixelTransformation& pixelTransformation,
        const InterpolationTransformation& interpolationTransformation,
        const Image& img,
        int numAttacks) {
    vector<pair<PointD, Image>> ret;
    std::default_random_engine generator;
    vector<PointD> randomParams = combinedDomain.sample(numAttacks, generator);
    cout << "Generating attacks..." << endl;
    for (const PointD& params : randomParams) {
        Image newImage(img.nRows, img.nCols, img.nChannels);
        for (int r = 0; r < img.nRows; ++r) {
            for (int c = 0; c < img.nCols; ++c) {
                for (int i = 0; i < img.nChannels; ++i) {
                    auto pixel = img.getPixel(r, c, i);
                    auto fLower = getLipschitzFunction(
                            img, pixel, combinedDomain,
                            spatialTransformation, pixelTransformation, interpolationTransformation, true);
                    auto fUpper = getLipschitzFunction(
                            img, pixel, combinedDomain,
                            spatialTransformation, pixelTransformation, interpolationTransformation, false);
                    newImage.a[r][c][i] = {fLower.f(params), fUpper.f(params)};
                }
            }
        }
        for (double param : params.x) {
		  out << param << endl;
        }
        newImage.print_csv(out);
        //newImage.print_ascii();
        ret.emplace_back(params, newImage);
    }
    cout << "Attacks generated!" << endl;
    return ret;
}

bool checkImagePoly(const Image& img, vector<Polyhedra> polys, PointD params) {
    assert((int)polys.size() == img.nRows * img.nCols * img.nChannels);
    int nxt = 0;
    for (size_t r = 0; r < img.nRows; ++r) {
        for (size_t c = 0; c < img.nCols; ++c) {
            for (size_t i = 0; i < img.nChannels; ++i) {
                Polyhedra poly = polys[nxt++];
                Interval polyEval = poly.evaluate(params);
                if (polyEval.inf > img.a[r][c][i].inf + Constants::EPS) return false;
                if (img.a[r][c][i].sup > polyEval.sup + Constants::EPS) return false;
            }
        }
    }
    return true;
}

bool checkImageBox(const Image& img, const Image& abstractImg) {
    for (size_t r = 0; r < img.nRows; ++r) {
        for (size_t c = 0; c < img.nCols; ++c) {
            for (size_t i = 0; i < img.nChannels; ++i) {
                if (img.a[r][c][i].inf + Constants::EPS < abstractImg.a[r][c][i].inf) return false;
                if (img.a[r][c][i].sup > abstractImg.a[r][c][i].sup + Constants::EPS) return false;
            }
        }
    }
    return true;
}

void sanityChecks(vector<bool> checked, vector<bool> checkedNumeric, vector<bool> checkedPoly, string calcType) {
    for (size_t i = 0; i < checked.size(); ++i) {
        assert(checked[i]);
        if (calcType == "baseline") {
            continue;
        }
        if (calcType == "polyhedra" || calcType == "custom_dp") {
            assert(checkedPoly[i]);
        }
    }
    cout << "Sanity checks passed!" << endl;
}

void getSplitPoints(vector<vector<double>>& splitPoints, string value) {
    string token;

    int idx = 0;
    vector<double> v;
    size_t pos = 0;

    while ((pos = value.find(',')) != string::npos) {
        token = value.substr(0, pos);
        if (token == "*") {
            cout << "pushing vector in" << endl;
            splitPoints.push_back(v);
            ++idx;
            v.clear();
        } else {
            cout << "pushing token to vector: " << stod(token) << endl;
            v.push_back(stod(token));
        }
        value.erase(0, pos + 1);
    }
    v.push_back(stod(value));
    splitPoints.push_back(v);
}

int main(int argc, char** argv) {
    assert(argc == 2);
    string out_dir = argv[1];

    double noise = 0;
    int n_splits = 1, inside_splits = 1;
    string calc_type = "baseline", dataset = "mnist", transformName, pixelTransformName;
    int numTests = 1;
    bool debug = false;
    string name, value;
    vector<vector<double>> splitPoints;
    string set = "test";

    ifstream config(out_dir + "/config.txt");
    while (config >> name >> value) {
        if (name == "" && value == "") {
            continue;
        }
        cout << "Setting property " << name << " to value: " << value << endl;
	if (name == "set") {
	    set = value;
	} else if (name == "split_points") {
            getSplitPoints(splitPoints, value);
            cout << "Total split points: " << splitPoints.size() << endl;
        } else if (name == "num_threads") {
            Constants::NUM_THREADS = stoi(value);
        } else if (name == "max_coeff") {
            Constants::MAX_COEFF = stod(value);
        } else if (name == "lp_samples") {
            Constants::LP_SAMPLES = stoi(value);
        } else if (name == "num_poly_check") {
            Constants::NUM_POLY_CHECK = stoi(value);
        } else if (name == "dataset") {
            dataset = value;
        } else if (name == "noise") {
            noise = stod(value);
        } else if (name == "chunks") {
            n_splits = stoi(value);
        } else if (name == "inside_splits") {
            inside_splits = stoi(value);
        } else if (name == "method"){
            calc_type = value;
        } else if (name == "spatial_transform") {
            transformName = value;
        } else if (name == "pixel_transform") {
            pixelTransformName = value;
        } else if (name == "num_tests") {
            numTests = stoi(value);
        } else if (name == "debug") {
            debug = true;
        } else if (name == "num_attacks") {
            Constants::NUM_ATTACKS = stoi(value);
        } else if (name == "poly_degree") {
            Constants::POLY_DEGREE = stoi(value);
        } else if (name == "poly_eps") {
            Constants::POLY_EPS = stod(value);
        } else if (name == "split_mode") {
            Constants::SPLIT_MODE = value;
        } else if (name == "ub_estimate") {
            Constants::UB_ESTIMATE = value;
        } else {
            cout << "Property not found: " << name << endl;
            return 1;
        }
    }
    
    assert(dataset == "mnist" || dataset == "fashion" || dataset == "cifar10" || dataset == "imagenet");
    SpatialTransformation& spatialTransformation = *getSpatialTransformation(transformName);
    PixelTransformation& pixelTransformation = *getPixelTransformation(pixelTransformName);
    const InterpolationTransformation& interpolationTransformation = InterpolationTransformation();
    string images = "datasets/" + dataset + "_" + set + ".csv";

	int nRows, nCols;
	if (dataset == "cifar10") {
	  nRows = nCols = 32;
	} else if (dataset == "mnist" || dataset == "fashion") {
	  nRows = nCols = 28;
	} else if (dataset == "imagenet") {
	  nRows = nCols = 250;
	} else {
	  assert(false);
	}

	cout << "nRows: " << nRows << ", nCols: " << nCols << endl;

    int nChannels = (dataset == "mnist" || dataset == "fashion") ? 1 : 3;
    double totalPolyRuntime = 0, totalBoxRuntime = 0;

    ifstream fin(images);
    vector<Image> imgs;
    string line;
    while (getline(fin, line) && (int)imgs.size() < numTests) {
        Image img = Image(nRows, nCols, nChannels, line, noise);
        imgs.push_back(img);
    }

    HyperBox combinedDomain = HyperBox::concatenate(spatialTransformation.domain, pixelTransformation.domain);

    auto verificationChunks = combinedDomain.split(n_splits, splitPoints);
    if (debug) {
        cout << "All verification chunks:" << endl;
        for (HyperBox& hbox : verificationChunks) {
            cout << "hbox: " << hbox << endl;
        }
    }

    std::vector<int> counts_picture;
    // iteration over images
    Statistics counter;

    for (size_t j = 0; j < std::min(numTests, (int)imgs.size()); ++j) {
        cout << "Image #" << j << endl;
        string out_file = out_dir + "/" + to_string(j) + ".csv";
        string attack_file = out_dir + "/attack_" + to_string(j) + ".csv";

        ofstream fou(out_file);
        ofstream fattack(attack_file);

        fou.precision(12);
        fattack.precision(12);

        fou.setf(ios_base::fixed);
        fattack.setf(ios_base::fixed);

        // auto attacks = generateAttacks(
        //         fattack, combinedDomain, spatialTransformation, pixelTransformation,
        //         interpolationTransformation, imgs[j], Constants::NUM_ATTACKS);
        // vector<bool> checked(attacks.size(), false);
        // vector<bool> checkedPoly(attacks.size(), false);
        // vector<bool> checkedNumeric(attacks.size(), false);

        imgs[j].print_csv(fou);
        fou << imgs[j].nChannels << " " << imgs[j].nRows << " " << imgs[j].nCols << endl;
        fou << combinedDomain.dim << endl;

        for (const HyperBox &hbox : verificationChunks) {
            for (const auto& it : hbox.it) {
                fou << it.inf << " " << it.sup << endl;
            }

            HyperBox hboxSpatial, hboxPixel;
            hbox.split(spatialTransformation.dim, hboxSpatial, hboxPixel);
            spatialTransformation.domain = hboxSpatial;
            pixelTransformation.domain = hboxPixel;

            // std::chrono::system_clock::time_point beginBox = std::chrono::system_clock::now();

            // cout << "Chunk: " << hbox << endl;
            // cout << "Interval box: " << endl;
            Image transformedImage = abstractWithSimpleBox(
                     hbox, imgs[j], spatialTransformation, pixelTransformation,
                     interpolationTransformation, inside_splits);

            // transformedImage.print_ascii();
            transformedImage.print_csv(fou);

            // std::chrono::system_clock::time_point endBox = std::chrono::system_clock::now();
            // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endBox - beginBox);
            // cout << "Box runtime (sec): " << duration.count() / 1000.0 << endl;
            // totalBoxRuntime += duration.count() / 1000.0;

            // for (size_t i = 0; i < attacks.size(); ++i) {
            //     if (hbox.inside(attacks[i].first) && checkImageBox(attacks[i].second, transformedImage)) {
            //         checked[i] = true;
            //     }
            // }

            // if (calc_type == "polyhedra") {
            //     cout << "Abstracting with DeepG" << endl;
            //     std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();

            //     vector<Polyhedra> polys = abstractWithPolyhedra(
            //             hbox, env, Constants::POLY_DEGREE, Constants::POLY_EPS, imgs[j],
            //             spatialTransformation, pixelTransformation, interpolationTransformation,
            //             transformedImage, counter);
            //     for (const auto &poly : polys) {
            //         fou << poly << endl;
            //     }

            //     for (size_t i = 0; i < attacks.size(); ++i) {
            //         if (hbox.inside(attacks[i].first) && checkImagePoly(attacks[i].second, polys, attacks[i].first)) {
            //             checkedPoly[i] = true;
            //         }
            //     }
            //     std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
            //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
            //     cout << "Poly runtime (sec): " << duration.count() / 1000.0 << endl;
            //     totalPolyRuntime += duration.count() / 1000.0;
            // } else if (calc_type == "custom_dp") {
            //     assert(hbox.dim == 1);
            //     vector<Polyhedra> polys = abstractWithCustomDP(
            //             hbox, imgs[j], spatialTransformation, interpolationTransformation, transformedImage);
            //     for (const auto &poly : polys) {
            //         fou << poly << endl;
            //     }
            //     for (size_t i = 0; i < attacks.size(); ++i) {
            //         if (hbox.inside(attacks[i].first) && checkImagePoly(attacks[i].second, polys, attacks[i].first)) {
            //             checkedPoly[i] = true;
            //         }
            //     }
            // }
            fou << "SPEC_FINISHED" << endl;
            counts_picture.push_back(counter.total_counts());
        }

        // sanityChecks(checked, checkedNumeric, checkedPoly, calc_type);
    }

    string stats_file = out_dir + "/log.txt";
    ofstream fstats(stats_file);

    fstats << "Avg poly runtime (s): " << totalPolyRuntime / (double)(numTests) << std::endl;
    fstats << "Avg box runtime (s): " << totalBoxRuntime / (double)(numTests) << std::endl;
    fstats << "Avg polyhedra distance: " << counter.getAveragePolyhedra() << std::endl;

    if (debug) {
        std::cout << "Counts: " << std::endl;
        for (const int &k : counts_picture) {
            std::cout << k << std::endl;
        }
    }

    return 0;
}
