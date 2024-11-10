from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key, needed for session management


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # Generate a random dataset X of size N with values between 0 and 1
    X = np.random.uniform(0, 1, N)

    # Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    # Y = beta0 + beta1 * X + mu + error term
    error = np.random.normal(0, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + mu + error

    # Fit a linear regression model to X and Y
    X_reshaped = X.reshape(-1, 1)
    model = LinearRegression().fit(X_reshaped, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X_reshaped), color='red', label=f'Fitted Line: y = {intercept:.2f} + {slope:.2f}x')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot and Regression Line')
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        # Generate simulated datasets using the same beta0 and beta1
        X_sim = np.random.uniform(0, 1, N)
        error_sim = np.random.normal(0, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim + mu + error_sim

        # Fit linear regression to simulated data and store slope and intercept
        X_sim_reshaped = X_sim.reshape(-1, 1)
        sim_model = LinearRegression().fit(X_sim_reshaped, Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    # Plot histograms of slopes and intercepts (matching previous assignment)
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color='blue', label='Slopes')
    plt.hist(intercepts, bins=20, alpha=0.5, color='orange', label='Intercepts')
    plt.axvline(slope, color='blue', linestyle='--', linewidth=1, label=f'Observed Slope: {slope:.2f}')
    plt.axvline(intercept, color='orange', linestyle='--', linewidth=1, label=f'Observed Intercept: {intercept:.2f}')
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Return data needed for further analysis, including slopes and intercepts
    # Calculate proportions of slopes and intercepts more extreme than observed
    slopes_array = np.array(slopes)
    intercepts_array = np.array(intercepts)

    slope_more_extreme = np.mean(np.abs(slopes_array - beta1) >= np.abs(slope - beta1))
    intercept_more_extreme = np.mean(np.abs(intercepts_array - beta0) >= np.abs(intercept - beta0))

    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_more_extreme,
        slopes,
        intercepts,
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # TODO 10: Calculate p-value based on test type
    if test_type == "greater":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "less":
        p_value = np.mean(simulated_stats <= observed_stat)
    else:  # not equal
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))

    # TODO 11: If p_value is very small (e.g., <= 0.0001), set fun_message to a fun message
    if p_value <= 0.0001:
        fun_message = "Wow! You've found a statistically significant result!"
    else:
        fun_message = None

    # TODO 12: Plot histogram of simulated statistics
    plt.figure()
    plt.hist(simulated_stats, bins=20, color='lightgreen', edgecolor='black', alpha=0.7, label='Simulated Statistics')
    plt.axvline(observed_stat, color='red', linestyle='--', linewidth=2, label=f'Observed {parameter.capitalize()}: {observed_stat:.4f}')
    plt.axvline(hypothesized_value, color='blue', linestyle='-', linewidth=2, label=f'Hypothesized {parameter.capitalize()} (H₀): {hypothesized_value}')
    plt.xlabel(parameter.capitalize())
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Simulated {parameter.capitalize()}s')
    plt.legend()
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )


@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level")) / 100  # Convert percentage to proportion

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    # Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)
    from scipy import stats
    # Calculate confidence interval for the parameter estimate using the t-distribution
    t_crit = stats.t.ppf((1 + confidence_level) / 2, df=S - 1)
    margin_of_error = t_crit * (std_estimate / np.sqrt(S))
    ci_lower = mean_estimate - margin_of_error
    ci_upper = mean_estimate + margin_of_error

    # Check if confidence interval includes true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    # Set x-axis limits to include the full range of estimates
    x_min = min(estimates) - 0.1 * abs(min(estimates))  # 10% buffer for visual clarity
    x_max = max(estimates) + 0.1 * abs(max(estimates))  # 10% buffer for visual clarity

    # Plot the individual estimates as points along the x-axis
    plt.figure(figsize=(10, 5))  # Adjusted height for a larger plot
    plt.scatter(estimates, [0] * len(estimates), color='gray', alpha=0.5, label='Simulated Estimates')
    mean_color = 'green' if includes_true else 'red'
    
    # Plot the mean estimate
    plt.scatter(mean_estimate, 0, color=mean_color, label=f'Mean Estimate: {mean_estimate:.4f}')
    
    # Plot the confidence interval as a single horizontal line
    plt.hlines(y=0, xmin=ci_lower, xmax=ci_upper, colors='blue', linestyles='-', label=f'{confidence_level*100:.0f}% Confidence Interval')
    
    # Plot the true parameter as a dotted vertical line
    plt.axvline(true_param, color='purple', linestyle=':', linewidth=2, label=f'True {parameter.capitalize()}: {true_param}')

    # Set the x-axis limits to capture all estimates
    plt.xlim(x_min, x_max)
    plt.xlabel(f'Estimated {parameter.capitalize()}')
    plt.title(f'Confidence Interval for {parameter.capitalize()}')
    plt.yticks([])  # Remove y-axis ticks
    plt.legend(loc="upper left")
    plot4_path = "static/plot4.png"
    plt.savefig(plot4_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level * 100,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )
if __name__ == "__main__":
    app.run(debug=True)