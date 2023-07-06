VERSION = (0, 0, 7)
PRERELEASE = None  # alpha, beta or rc
REVISION = None


def generate_version(version, prerelease=None, revision=None):
    version_parts = [".".join(map(str, version))]
    if prerelease is not None:
        version_parts.append("-{}".format(prerelease))
    if revision is not None:
        version_parts.append(".{}".format(revision))
    return "".join(version_parts)


__title__ = "pypipo"
__description__ = "Image convert to PIPO painting canvas automatically."
__url__ = "https://github.com/AutoPipo/pypipo/"
__version__ = generate_version(VERSION, prerelease=PRERELEASE, revision=REVISION)
__author__ = "Minku-Koo", "Jiyong-Park"
__author_email__ = "corleone@kakao.com"
__license__ = "MIT License"