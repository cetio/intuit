module tests.auc.tools;

import intuit.tool;
import std.math : exp, log;

/**
 * Pharmacokinetic prediction tools for tramadol concentration.
 * 
 * DATA SOURCES:
 * - DailyMed (NIH): Tramadol Hydrochloride Tablets Full Prescribing Information
 *   https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=a7f11004-b6fa-4c12-abe5-68909b352f04
 * - PubMed: Pharmacokinetics of tramadol and bioavailability of enteral tramadol formulations
 *   https://pubmed.ncbi.nlm.nih.gov/9793614/
 * 
 * PHARMACOKINETIC PARAMETERS:
 * - Bioavailability (F): 0.75 (75%) - DailyMed
 * - Volume of distribution (Vd): 2.6 L/kg (male), 2.9 L/kg (female) - DailyMed
 *   Using average: 2.75 L/kg
 * - Elimination half-life (t1/2): 6.3 ± 1.4 hours (tramadol) - DailyMed
 * - Tmax: 2 hours (peak concentration time) - DailyMed
 * - Linear pharmacokinetics confirmed - DailyMed
 * 
 * CALCULATED PARAMETERS:
 * - Elimination rate constant (k) = ln(2) / t1/2 = 0.693 / 6.3 = 0.11 h^-1
 * - Absorption rate constant (ka) ≈ 1.4 h^-1 (solved from Tmax equation)
 * 
 * MODEL:
 * One-compartment model with first-order absorption and elimination
 * C(t) = (F * Dose * ka / (Vd * (ka - k))) * (e^(-k*t) - e^(-ka*t))
 */

// Pharmacokinetic constants for tramadol
immutable double BIOAVAILABILITY = 0.75;  // 75% absolute bioavailability (DailyMed)
immutable double VD_PER_KG = 2.75;         // Volume of distribution in L/kg (average of 2.6 male, 2.9 female)
immutable double HALF_LIFE = 6.3;         // Elimination half-life in hours (DailyMed)
immutable double ELIMINATION_RATE = log(2.0) / HALF_LIFE;  // k = 0.11 h^-1
immutable double ABSORPTION_RATE = 1.4;    // ka = 1.4 h^-1 (calculated from Tmax=2h)

/**
 * Predict tramadol plasma concentration at a specified time after administration.
 * 
 * Parameters:
 *   time_hours - Time after administration in hours (must be >= 0)
 *   body_weight_kg - Patient body weight in kg (must be > 0)
 *   dose_mg - Dose in mg (must be > 0, default 50.0 for standard immediate-release tablet)
 * 
 * Returns:
 *   JSON object with concentration in ng/mL and metadata
 *   Returns error message for invalid inputs
 * 
 * Example usage:
 *   predict_tramadol_concentration(2.0, 70.0, 50.0)
 *   Returns: {"concentration_ng_per_ml": 156.2, "time_hours": 2.0, "dose_mg": 50.0, "body_weight_kg": 70.0}
 */
string predict_tramadol_concentration(double time_hours, double body_weight_kg, double dose_mg = 50.0)
{
    import std.format : format;
    
    // Input validation
    if (time_hours < 0.0 || body_weight_kg <= 0.0 || dose_mg <= 0.0)
    {
        return format(`{"error": "Invalid input parameters. ` ~
            `time_hours must be >= 0, body_weight_kg > 0, dose_mg > 0"}`);
    }
    
    // Calculate volume of distribution for this patient
    double vd = VD_PER_KG * body_weight_kg;  // L
    
    // One-compartment model with first-order absorption and elimination
    // C(t) = (F * Dose * ka / (Vd * (ka - k))) * (e^(-k*t) - e^(-ka*t))
    // Result in mg/L
    
    double numerator = BIOAVAILABILITY * dose_mg * ABSORPTION_RATE;
    double denominator = vd * (ABSORPTION_RATE - ELIMINATION_RATE);
    double exponential_term = exp(-ELIMINATION_RATE * time_hours) - exp(-ABSORPTION_RATE * time_hours);
    
    double concentration_mg_per_L = (numerator / denominator) * exponential_term;
    
    // Convert mg/L to ng/mL (1 mg/L = 1000 ng/mL)
    double concentration_ng_per_mL = concentration_mg_per_L * 1000.0;
    
    // Return result as JSON string
    return format(`{
        "concentration_ng_per_ml": %s,
        "time_hours": %s,
        "dose_mg": %s,
        "body_weight_kg": %s,
        "model": "one-compartment with first-order absorption and elimination",
        "bioavailability": %s,
        "volume_distribution_per_kg": %s,
        "half_life_hours": %s,
        "absorption_rate_per_hour": %s,
        "elimination_rate_per_hour": %s
    }`, concentration_ng_per_mL, time_hours, dose_mg, body_weight_kg,
        BIOAVAILABILITY, VD_PER_KG, HALF_LIFE, ABSORPTION_RATE, ELIMINATION_RATE);
}

/**
 * Get pharmacokinetic parameters for tramadol.
 * 
 * Returns:
 *   JSON object with all pharmacokinetic parameters used in the model
 * 
 * Example usage:
 *   get_tramadol_pharmacokinetics()
 *   Returns: {"bioavailability": 0.75, "vd_per_kg": 2.75, "half_life": 6.3, ...}
 */
string get_tramadol_pharmacokinetics()
{
    import std.format : format;
    
    return format(`{
        "bioavailability": %s,
        "volume_distribution_per_kg": %s,
        "half_life_hours": %s,
        "tmax_hours": 2.0,
        "absorption_rate_per_hour": %s,
        "elimination_rate_per_hour": %s,
        "protein_binding_percent": 20,
        "linear_pharmacokinetics": true,
        "active_metabolite": "M1 (O-desmethyltramadol)",
        "metabolite_half_life_hours": 7.4,
        "data_source": "DailyMed (NIH) and PubMed peer-reviewed studies"
    }`, BIOAVAILABILITY, VD_PER_KG, HALF_LIFE, ABSORPTION_RATE, ELIMINATION_RATE);
}

