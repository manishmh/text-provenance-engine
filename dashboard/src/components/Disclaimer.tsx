/* Fixed-scope disclaimer: robustness against the tested deterministic
 * transforms must never be read as resistance to paraphrasing or
 * adversarial attacks. Rendered on every robustness-related view. */

export function RobustnessDisclaimer() {
  return (
    <div className="alert alert--warning" data-testid="robustness-disclaimer">
      <strong>Measurement only.</strong> These results describe detector behavior
      under the tested deterministic transformations. They do not establish
      resistance to paraphrasing, watermark removal, or adversarial attacks,
      and a high robustness score does not mean the watermark is resistant
      to removal.
    </div>
  );
}
