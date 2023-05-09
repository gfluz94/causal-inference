import pandas as pd
import pytest


@pytest.fixture(scope="module")
def dummy_ate_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "T": {
                0: 0,
                1: 0,
                2: 1,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 1,
                8: 1,
                9: 0,
                10: 0,
                11: 0,
                12: 1,
                13: 1,
                14: 0,
                15: 0,
                16: 0,
                17: 0,
                18: 0,
                19: 1,
                20: 1,
                21: 1,
                22: 0,
                23: 0,
                24: 0,
                25: 0,
                26: 0,
                27: 0,
                28: 1,
                29: 0,
            },
            "X1": {
                0: 760.8871,
                1: 4738.984,
                2: 334.0493,
                3: 2719.5,
                4: 1367.806,
                5: 320.4677,
                6: 4096.258,
                7: 0.0,
                8: 0.0,
                9: 4056.871,
                10: 0.0,
                11: 39.3871,
                12: 0.0,
                13: 0.0,
                14: 1036.597,
                15: 841.4516,
                16: 6942.871,
                17: 2239.694,
                18: 0.0,
                19: 0.0,
                20: 0.0,
                21: 10941.35,
                22: 152.1774,
                23: 2792.903,
                24: 0.0,
                25: 1822.548,
                26: 0.0,
                27: 608.7097,
                28: 4398.95,
                29: 7565.903,
            },
            "X2": {
                0: 0,
                1: 0,
                2: 1,
                3: 0,
                4: 1,
                5: 1,
                6: 0,
                7: 1,
                8: 1,
                9: 0,
                10: 0,
                11: 0,
                12: 1,
                13: 1,
                14: 0,
                15: 1,
                16: 1,
                17: 0,
                18: 1,
                19: 1,
                20: 1,
                21: 1,
                22: 0,
                23: 0,
                24: 0,
                25: 1,
                26: 0,
                27: 0,
                28: 1,
                29: 0,
            },
            "Y": {
                0: 2340.719,
                1: 12705.49,
                2: 0.0,
                3: 0.0,
                4: 33.98771,
                5: 1273.8,
                6: 17358.85,
                7: 0.0,
                8: 8472.158,
                9: 5937.505,
                10: 3151.991,
                11: 6280.338,
                12: 0.0,
                13: 0.0,
                14: 0.0,
                15: 7933.914,
                16: 461.0507,
                17: 1892.968,
                18: 1730.418,
                19: 0.0,
                20: 8881.665,
                21: 15952.6,
                22: 10301.23,
                23: 6098.578,
                24: 1495.459,
                25: 803.8833,
                26: 0.0,
                27: 0.0,
                28: 0.0,
                29: 2838.713,
            },
        }
    )


@pytest.fixture(scope="module")
def dummy_ate_data_for_instrumental_variables() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Y": {
                0: 6.254442,
                1: 6.401885,
                2: 5.741831,
                3: 5.860403,
                4: 6.083491,
                5: 6.001272,
                6: 6.2148,
                7: 5.204112,
                8: 6.047781,
                9: 5.035328,
                10: 5.927259,
                11: 6.422404,
                12: 5.38705,
                13: 5.847161,
                14: 5.942446,
                15: 6.675314,
                16: 5.952494,
                17: 5.521845,
                18: 5.664895,
                19: 5.790019,
                20: 6.948677,
                21: 5.943455,
                22: 4.886213,
                23: 5.729413,
                24: 6.277041,
                25: 5.154292,
                26: 5.251619,
                27: 6.827817,
                28: 5.664895,
                29: 6.453171,
            },
            "T": {
                0: 16.0,
                1: 12.0,
                2: 12.0,
                3: 14.0,
                4: 12.0,
                5: 9.0,
                6: 14.0,
                7: 8.0,
                8: 12.0,
                9: 10.0,
                10: 12.0,
                11: 17.0,
                12: 7.0,
                13: 8.0,
                14: 9.0,
                15: 12.0,
                16: 13.0,
                17: 12.0,
                18: 14.0,
                19: 13.0,
                20: 12.0,
                21: 18.0,
                22: 7.0,
                23: 8.0,
                24: 12.0,
                25: 7.0,
                26: 8.0,
                27: 12.0,
                28: 12.0,
                29: 12.0,
            },
            "Xnum": {
                0: 38.0,
                1: 30.0,
                2: 32.0,
                3: 38.0,
                4: 38.0,
                5: 35.0,
                6: 38.0,
                7: 36.0,
                8: 33.0,
                9: 33.0,
                10: 35.0,
                11: 36.0,
                12: 30.0,
                13: 33.0,
                14: 31.0,
                15: 35.0,
                16: 34.0,
                17: 37.0,
                18: 38.0,
                19: 32.0,
                20: 33.0,
                21: 39.0,
                22: 35.0,
                23: 31.0,
                24: 34.0,
                25: 37.0,
                26: 35.0,
                27: 36.0,
                28: 35.0,
                29: 34.0,
            },
            "Z": {
                0: 0.0,
                1: 0.0,
                2: 1.0,
                3: 0.0,
                4: 0.0,
                5: 0.0,
                6: 1.0,
                7: 1.0,
                8: 1.0,
                9: 0.0,
                10: 1.0,
                11: 0.0,
                12: 0.0,
                13: 0.0,
                14: 1.0,
                15: 0.0,
                16: 0.0,
                17: 0.0,
                18: 0.0,
                19: 0.0,
                20: 0.0,
                21: 0.0,
                22: 0.0,
                23: 0.0,
                24: 0.0,
                25: 0.0,
                26: 0.0,
                27: 1.0,
                28: 1.0,
                29: 0.0,
            },
            "Xcat": {
                0: 29,
                1: 20,
                2: 20,
                3: 29,
                4: 20,
                5: 29,
                6: 20,
                7: 29,
                8: 20,
                9: 20,
                10: 20,
                11: 29,
                12: 29,
                13: 20,
                14: 20,
                15: 20,
                16: 29,
                17: 20,
                18: 29,
                19: 20,
                20: 29,
                21: 20,
                22: 29,
                23: 29,
                24: 20,
                25: 20,
                26: 29,
                27: 29,
                28: 20,
                29: 29,
            },
        }
    )


@pytest.fixture(scope="module")
def dummy_ate_data_for_propensity_score_matching() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Xcat": {
                3949: 0,
                4495: 0,
                7609: 1,
                6328: 0,
                7902: 0,
                3216: 1,
                1409: 0,
                5586: 1,
                2316: 0,
                5161: 0,
                736: 1,
                2751: 0,
                1252: 1,
                7319: 0,
                6022: 1,
                1823: 0,
                3331: 1,
                4509: 0,
                7445: 0,
                4354: 1,
                1198: 1,
                3155: 1,
                5492: 0,
                3597: 1,
                142: 1,
                1371: 1,
                936: 1,
                3591: 0,
                517: 1,
                6094: 1,
            },
            "Xnum": {
                3949: 37,
                4495: 31,
                7609: 73,
                6328: 37,
                7902: 70,
                3216: 52,
                1409: 53,
                5586: 44,
                2316: 39,
                5161: 56,
                736: 61,
                2751: 57,
                1252: 31,
                7319: 68,
                6022: 43,
                1823: 70,
                3331: 66,
                4509: 31,
                7445: 71,
                4354: 66,
                1198: 71,
                3155: 58,
                5492: 63,
                3597: 37,
                142: 50,
                1371: 78,
                936: 55,
                3591: 67,
                517: 55,
                6094: 63,
            },
            "Y": {
                3949: 14,
                4495: 12,
                7609: 10,
                6328: 12,
                7902: 21,
                3216: 12,
                1409: 15,
                5586: 35,
                2316: 12,
                5161: 11,
                736: 16,
                2751: 18,
                1252: 12,
                7319: 25,
                6022: 10,
                1823: 11,
                3331: 12,
                4509: 13,
                7445: 10,
                4354: 15,
                1198: 14,
                3155: 13,
                5492: 11,
                3597: 20,
                142: 10,
                1371: 10,
                936: 11,
                3591: 10,
                517: 34,
                6094: 10,
            },
            "T": {
                3949: 1,
                4495: 1,
                7609: 1,
                6328: 1,
                7902: 1,
                3216: 1,
                1409: 1,
                5586: 1,
                2316: 1,
                5161: 1,
                736: 1,
                2751: 1,
                1252: 1,
                7319: 1,
                6022: 1,
                1823: 0,
                3331: 0,
                4509: 0,
                7445: 0,
                4354: 0,
                1198: 0,
                3155: 0,
                5492: 0,
                3597: 0,
                142: 0,
                1371: 0,
                936: 0,
                3591: 0,
                517: 0,
                6094: 0,
            },
        }
    )
