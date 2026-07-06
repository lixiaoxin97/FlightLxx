import glob
import os
import sys


def ensure_libstdcxx_preloaded():
    candidates = [
        "/home/ubuntu/anaconda3/pkgs/libstdcxx-15.2.0-h39759b7_8/lib/libstdc++.so.6",
        "/home/ubuntu/anaconda3/pkgs/libstdcxx-15.2.0-h39759b7_7/lib/libstdc++.so.6",
        "/home/ubuntu/anaconda3/lib/libstdc++.so.6",
    ]
    candidates.extend(glob.glob("/home/ubuntu/anaconda3/pkgs/libstdcxx-*/lib/libstdc++.so.6"))

    for candidate in candidates:
        if os.path.exists(candidate):
            current_preload = os.environ.get("LD_PRELOAD", "")
            if candidate not in current_preload.split(":"):
                new_preload = candidate
                if current_preload:
                    new_preload += ":" + current_preload
                os.environ["LD_PRELOAD"] = new_preload
                os.environ["FLIGHTLXX_LIBSTDCXX_PRELOADED"] = "1"
                os.execv(sys.executable, [sys.executable] + sys.argv)
            return candidate

    return None
