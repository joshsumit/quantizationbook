// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="00_cover.html">Cover</a></li><li class="chapter-item expanded affix "><a href="00_title_page.html">Title Page</a></li><li class="chapter-item expanded affix "><a href="00_copyright.html">Copyright and License</a></li><li class="chapter-item expanded affix "><a href="00_preface.html">Preface</a></li><li class="chapter-item expanded affix "><li class="spacer"></li><li class="chapter-item expanded affix "><li class="part-title">Part I — Foundations</li><li class="chapter-item expanded "><a href="chapter01_why_quantization_exists.html"><strong aria-hidden="true">1.</strong> Why Quantization Exists</a></li><li class="chapter-item expanded "><a href="chapter02_quantization_as_representational_constraint.html"><strong aria-hidden="true">2.</strong> Quantization as a Representational Constraint</a></li><li class="chapter-item expanded "><a href="chapter03_scale_and_zero_point.html"><strong aria-hidden="true">3.</strong> Scale and Zero-Point</a></li><li class="chapter-item expanded "><a href="chapter04_hardware_dictates_the_rules.html"><strong aria-hidden="true">4.</strong> Hardware Dictates the Rules</a></li><li class="chapter-item expanded affix "><li class="part-title">Part II — The Error Model</li><li class="chapter-item expanded "><a href="chapter05_where_error_is_born.html"><strong aria-hidden="true">5.</strong> Where Error Is Born</a></li><li class="chapter-item expanded "><a href="chapter06_the_quantized_graph.html"><strong aria-hidden="true">6.</strong> The Quantized Graph</a></li><li class="chapter-item expanded "><a href="chapter07_requantization.html"><strong aria-hidden="true">7.</strong> Requantization</a></li><li class="chapter-item expanded "><a href="chapter08_operator_fusion.html"><strong aria-hidden="true">8.</strong> Operator Fusion</a></li><li class="chapter-item expanded affix "><li class="part-title">Part III — Calibration and Training</li><li class="chapter-item expanded "><a href="chapter09_calibration_and_observers.html"><strong aria-hidden="true">9.</strong> Calibration and Observers</a></li><li class="chapter-item expanded "><a href="chapter10_post_training_quantization.html"><strong aria-hidden="true">10.</strong> Post-Training Quantization</a></li><li class="chapter-item expanded "><a href="chapter11_quantization_aware_training.html"><strong aria-hidden="true">11.</strong> Quantization-Aware Training</a></li><li class="chapter-item expanded "><a href="chapter12_dynamic_and_mixed_precision.html"><strong aria-hidden="true">12.</strong> Dynamic Quantization and Mixed Precision</a></li><li class="chapter-item expanded affix "><li class="part-title">Part IV — Failure and Recovery</li><li class="chapter-item expanded "><a href="chapter13_failure_patterns.html"><strong aria-hidden="true">13.</strong> Failure Patterns</a></li><li class="chapter-item expanded "><a href="chapter14_why_transformers_break_quantization.html"><strong aria-hidden="true">14.</strong> Why Transformers Break Quantization</a></li><li class="chapter-item expanded "><a href="chapter15_smoothquant_and_outlier_control.html"><strong aria-hidden="true">15.</strong> SmoothQuant and Outlier Control</a></li><li class="chapter-item expanded affix "><li class="part-title">Part V — Modern Techniques</li><li class="chapter-item expanded "><a href="chapter16_weight_only_and_groupwise_quantization.html"><strong aria-hidden="true">16.</strong> Weight-Only and Group-Wise Quantization</a></li><li class="chapter-item expanded "><a href="chapter17_gptq_awq_and_offline_calibration.html"><strong aria-hidden="true">17.</strong> GPTQ, AWQ, and Offline Calibration</a></li><li class="chapter-item expanded "><a href="chapter18_the_kv_cache_bottleneck.html"><strong aria-hidden="true">18.</strong> The KV-Cache Bottleneck</a></li><li class="chapter-item expanded "><a href="chapter19_the_fp8_revolution.html"><strong aria-hidden="true">19.</strong> The FP8 Revolution</a></li><li class="chapter-item expanded affix "><li class="part-title">Part VI — Hardware Stacks</li><li class="chapter-item expanded "><a href="chapter20_qualcomm_stack.html"><strong aria-hidden="true">20.</strong> The Qualcomm Stack</a></li><li class="chapter-item expanded "><a href="chapter21_nvidia_stack.html"><strong aria-hidden="true">21.</strong> The NVIDIA Stack</a></li><li class="chapter-item expanded affix "><li class="spacer"></li><li class="chapter-item expanded affix "><li class="part-title">Appendices</li><li class="chapter-item expanded "><a href="appendix_quantization_toolkit_map.html"><strong aria-hidden="true">22.</strong> Appendix A: The Quantization Toolkit Map</a></li><li class="chapter-item expanded "><a href="appendix_end_to_end_walkthrough.html"><strong aria-hidden="true">23.</strong> Appendix B: End-to-End Numeric Walkthrough</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
