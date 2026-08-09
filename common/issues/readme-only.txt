# Recipes that are documentation only, so only README.md is required of them.
#
# One path per line, as <Checkpoint-Name>/<hardware>. Matched with grep -qxF, so the line must be the
# whole path with no leading "recipes/" and no trailing slash. Lines starting with # are ignored
# because they cannot match any recipe path.
#
# A recipe belongs here when it will never be launched as written. The repo carries working recipes
# only, so a blocked or never-launched configuration is removed rather than listed here, and its page is
# kept outside the repo.
#
# The list is empty, and by the rule above it should stay that way. The checks glob recipes/*/*/ and so
# do not enumerate model-level directories at all, so this file is documentation of intent rather than
# something the audit consults today. It is kept so the intent survives if the audit is later taught to
# walk model-level READMEs.
