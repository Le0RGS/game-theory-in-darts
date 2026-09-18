import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal


class Dartboard:
    """
    Represents a standard dartboard geometry and scoring rules.
    Dimensions are based on standard tournament specifications (mm).
    """

    def __init__(self):
        # Standard radii in millimeters
        self.R_BULLSEYE = 6.35
        self.R_OUTER_BULL = 15.9
        self.R_TREBLE_IN = 99.0
        self.R_TREBLE_OUT = 107.0
        self.R_DOUBLE_IN = 162.0
        self.R_DOUBLE_OUT = 170.0

        # Order of numbers starting from East (0 degrees),
        # counter-clockwise
        self.SECTORS = [
            6, 13, 4, 18, 1,
            20, 5, 12, 9, 14,
            11, 8, 16, 7, 19,
            3, 17, 2, 15, 10
        ]

    def get_score(self, x, y):
        """
        Calculate the score for a single coordinate (x, y)
        or arrays of coordinates.
        """

        r = np.sqrt(x**2 + y**2)
        theta = np.degrees(np.arctan2(y, x))
        theta = np.where(theta < 0, theta + 360, theta)

        is_bullseye = r < self.R_BULLSEYE
        is_outer_bull = (
            (r >= self.R_BULLSEYE)
            & (r < self.R_OUTER_BULL)
        )

        if np.isscalar(r):
            if is_bullseye:
                return 50
            if is_outer_bull:
                return 25
            if r > self.R_DOUBLE_OUT:
                return 0

        # Determine base sector score
        sector_index = ((theta + 9) // 18).astype(int) % 20
        base_scores = np.array(self.SECTORS)[sector_index]

        # Determine ring multiplier
        multiplier = np.ones_like(r)

        in_treble = (
            (r >= self.R_TREBLE_IN)
            & (r <= self.R_TREBLE_OUT)
        )
        multiplier[in_treble] = 3

        in_double = (
            (r >= self.R_DOUBLE_IN)
            & (r <= self.R_DOUBLE_OUT)
        )
        multiplier[in_double] = 2

        final_score = base_scores * multiplier

        if not np.isscalar(r):
            final_score[is_bullseye] = 50
            final_score[is_outer_bull] = 25
            final_score[r > self.R_DOUBLE_OUT] = 0

        return final_score


class Player:
    """
    Represents a player with a specific skill level,
    measured by sigma (standard deviation in mm).
    """

    def __init__(self, name, skill_sigma):
        self.name = name
        self.sigma = skill_sigma

    def throw_darts(self, aim_x, aim_y, n_throws=1):
        dev_x = np.random.normal(0, self.sigma, n_throws)
        dev_y = np.random.normal(0, self.sigma, n_throws)

        return aim_x + dev_x, aim_y + dev_y


class Simulation:
    def __init__(self, dartboard):
        self.board = dartboard

    def visualize_heatmap(self, player, resolution=2):
        print(f"Generating strategy heatmap for {player.name}...")

        extent = 180

        xs = np.arange(-extent, extent, resolution)
        ys = np.arange(-extent, extent, resolution)

        X, Y = np.meshgrid(xs, ys)

        flat_X = X.ravel()
        flat_Y = Y.ravel()

        exact_scores = self.board.get_score(
            flat_X, flat_Y
        ).reshape(X.shape)

        # Apply Gaussian filter to simulate inaccuracy
        sigma_pixels = player.sigma / resolution

        ev_map = gaussian_filter(
            exact_scores.astype(float),
            sigma=sigma_pixels
        )

        plt.figure(figsize=(10, 8))
        plt.title(
            f"Expected Payoff Heatmap\n"
            f"Player: {player.name}"
        )

        im = plt.imshow(
            ev_map,
            extent=[
                -extent,
                extent,
                -extent,
                extent
            ],
            origin="lower",
            cmap="inferno"
        )

        plt.colorbar(im, label="Expected Score")

        # Draw board rings
        circles = [
            self.board.R_BULLSEYE,
            self.board.R_OUTER_BULL,
            self.board.R_TREBLE_IN,
            self.board.R_TREBLE_OUT,
            self.board.R_DOUBLE_IN,
            self.board.R_DOUBLE_OUT
        ]

        ax = plt.gca()

        for r in circles:
            circle = patches.Circle(
                (0, 0),
                r,
                fill=False,
                color="white",
                alpha=0.3
            )
            ax.add_patch(circle)

        # Find optimal aim point
        max_idx = np.unravel_index(
            np.argmax(ev_map),
            ev_map.shape
        )

        best_y = ys[max_idx[0]]
        best_x = xs[max_idx[1]]

        plt.plot(
            best_x,
            best_y,
            "cx",
            markersize=10,
            markeredgewidth=2,
            label="Optimal Aim"
        )

        plt.legend()
        plt.show()


def get_bullseye_probability(sigma, r_bullseye=6.35):
    if sigma <= 0:
        return 1.0

    return 1 - np.exp(
        -(r_bullseye**2) / (2 * sigma**2)
    )


def integrate_gaussian_over_polar_sector(
    sigma,
    r_min,
    r_max,
    theta_center_deg,
    width_deg=18
):
    # Define aim point at centre of sector
    r_aim = (r_min + r_max) / 2.0
    theta_aim_rad = np.radians(theta_center_deg)

    aim_x = r_aim * np.cos(theta_aim_rad)
    aim_y = r_aim * np.sin(theta_aim_rad)

    # Numerical integration setup
    r_steps = 50
    theta_steps = 50

    rs = np.linspace(r_min, r_max, r_steps)

    thetas = np.linspace(
        np.radians(
            theta_center_deg - width_deg / 2
        ),
        np.radians(
            theta_center_deg + width_deg / 2
        ),
        theta_steps
    )

    R, T = np.meshgrid(rs, thetas)

    X = R * np.cos(T)
    Y = R * np.sin(T)

    rv = multivariate_normal(
        [aim_x, aim_y],
        [[sigma**2, 0], [0, sigma**2]]
    )

    pos = np.dstack((X, Y))
    pdf_values = rv.pdf(pos)

    dr = (r_max - r_min) / (r_steps - 1)
    dtheta = np.radians(width_deg) / (theta_steps - 1)

    return np.sum(
        pdf_values * R * dr * dtheta
    )


def get_hit_probability(sigma, target_type="T20"):
    board = Dartboard()
    type_code = target_type[0]

    if type_code == "D":
        r_min = board.R_DOUBLE_IN
        r_max = board.R_DOUBLE_OUT

    elif type_code == "T":
        r_min = board.R_TREBLE_IN
        r_max = board.R_TREBLE_OUT

    elif type_code == "S":
        r_min = board.R_TREBLE_OUT
        r_max = board.R_DOUBLE_IN

    else:
        raise ValueError("Unknown target type")

    return integrate_gaussian_over_polar_sector(
        sigma,
        r_min,
        r_max,
        0
    )


if __name__ == "__main__":

    board = Dartboard()
    sim = Simulation(board)

    # Skill levels
    sigma_pro = 10
    sigma_amateur = 40

    # Calculate probabilities for professional player
    p_treble = get_hit_probability(
        sigma_pro,
        "T"
    )

    p_double = get_hit_probability(
        sigma_pro,
        "D"
    )

    p_single = get_hit_probability(
        sigma_pro,
        "S"
    )

    p_bull = get_bullseye_probability(
        sigma_pro
    )

    p_next = p_double * p_single
    p_err = 0.05

    print(
        f"Amateur bullseye probability: "
        f"{get_bullseye_probability(sigma_amateur):.2%}"
    )

    print(
        f"\n--- Pro Player "
        f"(Sigma={sigma_pro}mm) ---"
    )

    print(
        f"Probability of hitting Treble: "
        f"{p_treble:.2%}"
    )

    print(
        f"Probability of hitting Double: "
        f"{p_double:.2%}"
    )

    print(
        f"Probability of hitting Single: "
        f"{p_single:.2%}"
    )

    print(
        f"Probability of hitting Bullseye: "
        f"{p_bull:.2%}"
    )

    # Generate expected payoff heatmaps
    pro_player = Player(
        "Pro Player",
        skill_sigma=sigma_pro
    )

    amateur_player = Player(
        "Amateur",
        skill_sigma=sigma_amateur
    )

    sim.visualize_heatmap(pro_player)
    sim.visualize_heatmap(amateur_player)

    # Strategic analysis
    q_opp = np.linspace(0, 1, 100)

    # Calculate win probabilities
    P_safe = (1 - q_opp) * p_next

    P_agg = (
        p_bull
        + (1 - p_bull) * (1 - q_opp) * p_err
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        q_opp,
        P_safe,
        label="Strategy Safe"
    )

    plt.plot(
        q_opp,
        P_agg,
        label="Strategy Aggressive",
        linestyle="--"
    )

    # Find intersection
    idx = np.argwhere(
        np.diff(
            np.sign(P_agg - P_safe)
        )
    ).flatten()

    if len(idx) > 0:
        intersect_x = q_opp[idx[0]]
        intersect_y = P_agg[idx[0]]

        plt.plot(
            intersect_x,
            intersect_y,
            "ko"
        )

        plt.annotate(
            f"Critical Point\n"
            f"q ≈ {intersect_x:.2f}",
            xy=(intersect_x, intersect_y),
            xytext=(
                intersect_x + 0.1,
                intersect_y + 0.15
            )
        )

    plt.title(
        "Strategic Indifference: "
        "Aggression vs. Safety"
    )

    plt.xlabel(
        "Opponent's Threat Level (q_opp)"
    )

    plt.ylabel(
        "Player A's Probability of Winning (P_win)"
    )

    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.xlim(0, 1)
    plt.ylim(0, 0.7)

    plt.savefig(
        "zero_sum_plot.png",
        dpi=300
    )

    plt.show()
