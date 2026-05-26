import type { UploadProgressState } from "../types";

const UPLOAD_PROGRESS_CAP = 88;
const UPLOAD_SERVER_PHASE = 94;

export function postFileWithProgress(
  file: File,
  folder: string,
  tags: string,
  fileIndex: number,
  totalFiles: number,
  onProgress: (state: UploadProgressState) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("folder", folder);
    fd.append("tags", tags);

    const xhr = new XMLHttpRequest();
    xhr.timeout = 600_000;
    const basePercent = (fileIndex / totalFiles) * 100;
    const slice = 100 / totalFiles;
    let bytesFullySent = false;

    const emit = (partial: Partial<UploadProgressState>) => {
      onProgress({
        visible: true,
        percent: 0,
        label: "",
        detail: "",
        failed: false,
        ...partial,
      });
    };

    xhr.upload.addEventListener("progress", (e) => {
      const loadedMb = (e.loaded / (1024 * 1024)).toFixed(1);
      const totalMb = e.lengthComputable
        ? (e.total / (1024 * 1024)).toFixed(1)
        : "?";

      if (e.lengthComputable && e.total > 0 && e.loaded >= e.total) {
        bytesFullySent = true;
        const overall = basePercent + (UPLOAD_SERVER_PHASE / 100) * slice;
        emit({
          percent: overall,
          label: `Saving ${file.name}…`,
          detail: `File ${fileIndex + 1} of ${totalFiles} · sent ${totalMb} MB, waiting for server`,
        });
        return;
      }

      let filePct = 50;
      if (e.lengthComputable && e.total > 0) {
        filePct = Math.min(UPLOAD_PROGRESS_CAP, (e.loaded / e.total) * 100);
      }
      const overall = basePercent + (filePct / 100) * slice;
      emit({
        percent: overall,
        label: `Uploading ${file.name}`,
        detail: `File ${fileIndex + 1} of ${totalFiles} · ${loadedMb} / ${totalMb} MB`,
      });
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const overall = basePercent + slice;
        emit({
          percent: overall,
          label: `Saved ${file.name}`,
          detail: bytesFullySent
            ? "Upload accepted — indexing runs in background"
            : `File ${fileIndex + 1} of ${totalFiles} saved`,
        });
        resolve();
        return;
      }
      let msg = `Upload failed (${xhr.status})`;
      try {
        const body = JSON.parse(xhr.responseText);
        if (body.detail) msg = body.detail;
      } catch {
        /* ignore */
      }
      reject(new Error(msg));
    });

    xhr.addEventListener("error", () =>
      reject(new Error("Network error during upload")),
    );
    xhr.addEventListener("abort", () => reject(new Error("Upload cancelled")));
    xhr.addEventListener("timeout", () =>
      reject(new Error("Upload timed out (server too slow)")),
    );

    xhr.open("POST", "/api/documents");
    xhr.send(fd);
  });
}
