/** `pyRadPlan/app/about.json`, embedded by `vite.config.ts`. */
declare const __PYRADPLAN_ABOUT__: {
  description: string;
  license_note: string;
  disclaimer_title: string;
  disclaimer: string;
};

interface ImportMetaEnv {
  /** Deployment channel ("latest", "develop"), set by `web/scripts/assemble_pages.mjs`. */
  readonly VITE_PYRADPLAN_CHANNEL?: string;
  /** `git describe` of the sources the build was made from. */
  readonly VITE_PYRADPLAN_VERSION?: string;
  /** Short commit hash of the sources. */
  readonly VITE_PYRADPLAN_COMMIT?: string;
}
