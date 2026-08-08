//! Sample Rust client for the FLUX image generator REST API (`/api/v1`).
//!
//! ```no_run
//! # async fn demo() -> Result<(), flux_client::Error> {
//! use flux_client::{FluxClient, GenerateRequest};
//!
//! let flux = FluxClient::from_env()?;
//! flux.wait_until_ready(None).await?;
//!
//! let job = flux
//!     .generate(&GenerateRequest::new("a red fox in falling snow").steps(30), None)
//!     .await?;
//!
//! for image in &job.images {
//!     println!("{} (seed {})", image.filename, image.seed);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Two properties of the API shape this client:
//!
//! * **Generation is always asynchronous.** `POST /jobs` returns 201 with an id
//!   even when the queue is empty, because one worker thread runs every job in
//!   turn. [`FluxClient::submit`] gives you the id; [`FluxClient::generate`]
//!   does the polling for you.
//! * **Errors carry a stable code.** Every failure is
//!   `{"error": {"code", "message"}}`. Match on [`Error::Api`]'s `code` rather
//!   than the message, which is written for humans and may change.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// States a job settles into. Anything else means it is still in flight.
pub const TERMINAL_STATES: [&str; 3] = ["done", "failed", "canceled"];

const API_PREFIX: &str = "/api/v1";

// ----------------------------------------------------------------- errors --

#[derive(Debug)]
pub enum Error {
    /// The server rejected the request. `code` is the stable identifier to
    /// match on: `model_loading`, `queue_full`, `not_found`, `busy`,
    /// `invalid_request`, `vlm_failed`, ...
    Api {
        code: String,
        message: String,
        status: u16,
    },
    /// The request never completed (connection refused, DNS, TLS, timeout).
    /// Expected transiently while the server restarts into another config.
    Transport(reqwest::Error),
    Io(std::io::Error),
    /// A wait helper gave up.
    Timeout(String),
    /// No API key available.
    MissingApiKey,
}

impl Error {
    /// The server-side error code, if this was a rejection rather than a
    /// transport failure.
    pub fn code(&self) -> Option<&str> {
        match self {
            Error::Api { code, .. } => Some(code),
            _ => None,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Api {
                code,
                message,
                status,
            } => write!(f, "{code} (HTTP {status}): {message}"),
            Error::Transport(e) => write!(f, "transport error: {e}"),
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Timeout(what) => write!(f, "timed out waiting for {what}"),
            Error::MissingApiKey => write!(
                f,
                "no API key: set FLUX_API_KEY to the key the server was started with"
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        Error::Transport(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    #[serde(default)]
    code: String,
    #[serde(default)]
    message: String,
}

// ------------------------------------------------------------------ types --

#[derive(Debug, Clone, Deserialize)]
pub struct Health {
    pub ready: bool,
    /// Narrates the load: "loading FLUX.2 model", "loading turbo LoRA", ...
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub elapsed_s: f64,
    #[serde(default)]
    pub server_version: Option<String>,
}

/// Which optional features the loaded backend can actually serve. Check these
/// before sending `negative_prompt`, `mask_image`, or several references —
/// an unsupported field is a 400, not a silent no-op.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub model: String,
    pub description: String,
    pub encoder: String,
    pub flux_version: u32,
    pub turbo: bool,
    pub schnell: bool,
    pub kontext: bool,
    pub uncensored: bool,
    pub sd: bool,
    pub negative_prompt: bool,
    pub inpaint: bool,
    #[serde(default)]
    pub hostname: String,
    #[serde(default)]
    pub version: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConfigEntry {
    pub id: u32,
    pub label: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConfigList {
    pub configs: Vec<ConfigEntry>,
    pub current: Option<u32>,
    /// False when the server was started without the supervisor, in which case
    /// it cannot restart itself into another config.
    pub switchable: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JobImage {
    pub filename: String,
    pub seed: i64,
    #[serde(default)]
    pub guidance: Option<f64>,
    #[serde(default)]
    pub strength: Option<f64>,
    #[serde(default)]
    pub timings: HashMap<String, f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Job {
    pub id: String,
    /// `queued` -> `running` -> `done` | `failed` | `canceled`.
    pub state: String,
    #[serde(default)]
    pub prompt: String,
    /// 1-based index of the image being produced within the batch.
    #[serde(default)]
    pub current: u32,
    #[serde(default)]
    pub batch: u32,
    #[serde(default)]
    pub step: u32,
    /// Re-read from the live scheduler on the first step: img2img, turbo and
    /// schnell denoise fewer steps than requested.
    #[serde(default)]
    pub total_steps: u32,
    #[serde(default)]
    pub images: Vec<JobImage>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub generation_time: f64,
    /// Only present on the response to `POST /jobs`.
    #[serde(default)]
    pub position: Option<u32>,
    #[serde(default)]
    pub preview: Option<String>,
    #[serde(default)]
    pub preview_step: u32,
}

impl Job {
    pub fn is_finished(&self) -> bool {
        TERMINAL_STATES.contains(&self.state.as_str())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct QueueSnapshot {
    pub running: Option<Job>,
    pub queued: Vec<serde_json::Value>,
    pub recent: Vec<Job>,
    pub queue_max_size: u32,
}

/// One per-step frame written to disk during generation.
#[derive(Debug, Clone, Deserialize)]
pub struct PreviewFrame {
    pub path: String,
    pub url: String,
    /// 1-based index of the batch image this frame belongs to.
    pub image: Option<u32>,
    pub step: Option<u32>,
}

/// The frame being denoised right now. Overwritten in place as generation
/// proceeds, so `url` carries `ts` as a cache-buster.
#[derive(Debug, Clone, Deserialize)]
pub struct LivePreview {
    pub path: String,
    pub url: String,
    pub step: u32,
    pub total_steps: u32,
    pub image: u32,
    pub ts: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JobPreviews {
    pub id: String,
    pub state: Option<String>,
    /// Present only while the job runs with `show_preview`.
    pub live: Option<LivePreview>,
    /// Written only with `save_previews`, but they persist after the job ends.
    pub frames: Vec<PreviewFrame>,
    pub count: u32,
    pub saving: bool,
}

/// A pending job, with its place in line.
#[derive(Debug, Clone, Deserialize)]
pub struct WaitingJob {
    pub id: String,
    /// 1-based place in the queue.
    pub position: u32,
    pub state: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default = "one")]
    pub batch: u32,
}

fn one() -> u32 {
    1
}

/// The queue-centric view from `GET /queue`.
#[derive(Debug, Clone, Deserialize)]
pub struct QueueView {
    pub running: Option<Job>,
    pub waiting: Vec<WaitingJob>,
    pub depth: u32,
    pub capacity: u32,
    /// False once `depth` reaches `capacity`: a further POST /jobs would be
    /// rejected with `queue_full`. The running job holds no pending slot, so a
    /// full queue can still have one in flight.
    pub accepting: bool,
    pub busy: bool,
    pub images_pending: u32,
    /// Null until at least one job has completed to learn from.
    pub seconds_per_image: Option<f64>,
    pub estimated_wait_s: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HistoryImage {
    pub filename: String,
    pub time: String,
    pub prompt: Option<String>,
}

/// What the reference-import endpoints return: a JPEG data URL bounded to
/// 2048px, ready to drop into [`GenerateRequest::input_images`].
#[derive(Debug, Clone, Deserialize)]
pub struct ImportedImage {
    pub image: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DirListing {
    pub dir: String,
    pub parent: Option<String>,
    pub dirs: Vec<String>,
    pub files: Vec<DirFile>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DirFile {
    pub filename: String,
}

#[derive(Debug, Clone, Deserialize)]
struct VlmStarted {
    id: String,
}

#[derive(Debug, Clone, Deserialize)]
struct VlmPoll {
    done: bool,
    #[serde(default)]
    result: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MultiRunStarted {
    pub id: String,
    pub configs: Vec<u32>,
    pub seed: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MultiRunState {
    pub active: bool,
    pub run: serde_json::Value,
    #[serde(default)]
    pub current_config: Option<u32>,
    #[serde(default)]
    pub next_config: Option<u32>,
}

// --------------------------------------------------------- request bodies --

/// A generation request. Only `prompt` is required; every other field is
/// omitted from the JSON unless set, so the server's defaults apply.
#[derive(Debug, Clone, Default, Serialize)]
pub struct GenerateRequest {
    pub prompt: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guidance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orientation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// SDXL backend only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// Base64 or data URLs, up to three. More than one needs Kontext or FLUX.2.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub input_images: Vec<String>,
    /// Paths the server itself can read — avoids uploading entirely.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub input_paths: Vec<String>,
    /// Inpainting mask; needs exactly one input image and a FLUX.2/SDXL backend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_image: Option<String>,
    /// "keep" derives the output dimensions from the reference's aspect ratio.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_preview: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub save_previews: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spectrum_grid: Option<bool>,
}

impl GenerateRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            ..Default::default()
        }
    }

    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = Some(steps);
        self
    }
    pub fn batch(mut self, batch: u32) -> Self {
        self.batch = Some(batch);
        self
    }
    pub fn seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }
    pub fn guidance(mut self, guidance: f64) -> Self {
        self.guidance = Some(guidance);
        self
    }
    pub fn strength(mut self, strength: f64) -> Self {
        self.strength = Some(strength);
        self
    }
    pub fn size(mut self, size: impl Into<String>) -> Self {
        self.size = Some(size.into());
        self
    }
    pub fn orientation(mut self, orientation: impl Into<String>) -> Self {
        self.orientation = Some(orientation.into());
        self
    }
    pub fn reference(mut self, data_url: impl Into<String>) -> Self {
        self.input_images.push(data_url.into());
        self
    }
    pub fn server_path(mut self, path: impl Into<String>) -> Self {
        self.input_paths.push(path.into());
        self
    }
    pub fn mask(mut self, data_url: impl Into<String>) -> Self {
        self.mask_image = Some(data_url.into());
        self
    }
    pub fn keep_aspect(mut self) -> Self {
        self.aspect_mode = Some("keep".into());
        self
    }
    pub fn with_previews(mut self) -> Self {
        self.show_preview = Some(true);
        self
    }
}

/// Read an image file into the data URL the API accepts for references.
pub fn encode_image(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let mime = match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "image/png",
    };
    let bytes = std::fs::read(path)?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(format!("data:{mime};base64,{b64}"))
}

// ----------------------------------------------------------------- client --

pub struct FluxClient {
    base: String,
    api_key: String,
    http: reqwest::Client,
}

impl FluxClient {
    pub fn new(base_url: &str, api_key: impl Into<String>) -> Result<Self> {
        let api_key = api_key.into();
        if api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        Ok(Self {
            base: format!("{}{API_PREFIX}", base_url.trim_end_matches('/')),
            api_key,
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()?,
        })
    }

    /// Read `FLUX_URL` (default `http://localhost:2222`) and `FLUX_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let url = std::env::var("FLUX_URL").unwrap_or_else(|_| "http://localhost:2222".into());
        let key = std::env::var("FLUX_API_KEY").map_err(|_| Error::MissingApiKey)?;
        Self::new(&url, key)
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    /// Send a request and decode the response, turning the API's error
    /// envelope into [`Error::Api`].
    async fn send<T: DeserializeOwned>(&self, req: reqwest::RequestBuilder) -> Result<T> {
        let response = req.header("X-API-Key", &self.api_key).send().await?;
        let status = response.status();
        let bytes = response.bytes().await?;

        if !status.is_success() {
            return Err(match serde_json::from_slice::<ErrorEnvelope>(&bytes) {
                Ok(env) => Error::Api {
                    code: env.error.code,
                    message: env.error.message,
                    status: status.as_u16(),
                },
                Err(_) => Error::Api {
                    code: "unparseable_response".into(),
                    message: String::from_utf8_lossy(&bytes).chars().take(200).collect(),
                    status: status.as_u16(),
                },
            });
        }

        // 204 No Content, and any other empty body, decode as JSON null so
        // callers can use `()` as the response type.
        if bytes.is_empty() {
            return serde_json::from_str("null").map_err(|e| Error::Api {
                code: "unparseable_response".into(),
                message: e.to_string(),
                status: status.as_u16(),
            });
        }

        serde_json::from_slice(&bytes).map_err(|e| Error::Api {
            code: "unparseable_response".into(),
            message: e.to_string(),
            status: status.as_u16(),
        })
    }

    async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        self.send(self.http.get(self.url(path))).await
    }

    async fn post_json<T: DeserializeOwned>(&self, path: &str, body: &impl Serialize) -> Result<T> {
        self.send(self.http.post(self.url(path)).json(body)).await
    }

    async fn post_empty<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        self.send(self.http.post(self.url(path))).await
    }

    async fn delete<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        self.send(self.http.delete(self.url(path))).await
    }

    // -- readiness and model ------------------------------------------------

    /// Readiness. Needs no key, and returns the body even on 503 ("still
    /// loading" is information, not a failure).
    pub async fn health(&self) -> Result<Health> {
        let response = self.http.get(self.url("/health")).send().await?;
        Ok(response.json().await?)
    }

    /// Block until the model is loaded. A cold 32B load takes minutes, and a
    /// config switch restarts the process — so transport errors are retried
    /// rather than propagated.
    pub async fn wait_until_ready(&self, on_status: Option<&dyn Fn(&str)>) -> Result<Health> {
        let deadline = Instant::now() + Duration::from_secs(1800);
        let mut last = String::new();
        while Instant::now() < deadline {
            match self.health().await {
                Ok(state) => {
                    if state.ready {
                        return Ok(state);
                    }
                    if let Some(err) = &state.error {
                        return Err(Error::Api {
                            code: "model_load_failed".into(),
                            message: err.clone(),
                            status: 503,
                        });
                    }
                    if state.status != last {
                        last = state.status.clone();
                        if let Some(cb) = on_status {
                            cb(&last);
                        }
                    }
                }
                Err(Error::Transport(_)) => {
                    if let Some(cb) = on_status {
                        cb("server unreachable (restarting?)");
                    }
                }
                Err(e) => return Err(e),
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
        Err(Error::Timeout("the model to load".into()))
    }

    pub async fn model(&self) -> Result<ModelInfo> {
        self.get("/model").await
    }

    pub async fn models(&self) -> Result<ConfigList> {
        self.get("/models").await
    }

    /// Switch config. This restarts the server process, so a 202 means the
    /// restart was *scheduled*; follow with [`Self::wait_until_ready`].
    pub async fn switch_model(&self, config: u32) -> Result<serde_json::Value> {
        self.send(
            self.http
                .put(self.url("/models/current"))
                .json(&serde_json::json!({ "config": config })),
        )
        .await
    }

    pub async fn telemetry(&self) -> Result<serde_json::Value> {
        self.get("/telemetry").await
    }

    // -- generation ---------------------------------------------------------

    /// Queue a job and return immediately.
    pub async fn submit(&self, request: &GenerateRequest) -> Result<Job> {
        self.post_json("/jobs", request).await
    }

    pub async fn job(&self, id: &str) -> Result<Job> {
        self.get(&format!("/jobs/{id}")).await
    }

    pub async fn jobs(&self) -> Result<QueueSnapshot> {
        self.get("/jobs").await
    }

    /// The current queue, ordered: what is generating, what is waiting and in
    /// what position, and roughly how long the backlog will take.
    pub async fn queue(&self) -> Result<QueueView> {
        self.get("/queue").await
    }

    /// Cancel a queued job, or interrupt the running one. Batch images that
    /// already finished are kept.
    pub async fn cancel(&self, id: &str) -> Result<serde_json::Value> {
        self.delete(&format!("/jobs/{id}")).await
    }

    /// Intermediate images from a generation: the frame being denoised right
    /// now (`live`, only with `show_preview`) plus any per-step frames written
    /// to disk (`frames`, only with `save_previews` — but they outlive the job).
    pub async fn previews(&self, id: &str) -> Result<JobPreviews> {
        self.get(&format!("/jobs/{id}/previews")).await
    }

    /// Poll until the job settles. `on_progress` fires whenever the
    /// (state, image, step) triple changes — enough to drive a progress bar.
    pub async fn wait_for_job(&self, id: &str, on_progress: Option<&dyn Fn(&Job)>) -> Result<Job> {
        let deadline = Instant::now() + Duration::from_secs(3600);
        let mut last = (String::new(), 0, 0);
        while Instant::now() < deadline {
            let job = self.job(id).await?;
            let key = (job.state.clone(), job.current, job.step);
            if key != last {
                last = key;
                if let Some(cb) = on_progress {
                    cb(&job);
                }
            }
            if job.is_finished() {
                return Ok(job);
            }
            tokio::time::sleep(Duration::from_millis(1500)).await;
        }
        Err(Error::Timeout(format!("job {id}")))
    }

    /// Submit and wait. Returns `Error::Api { code: "generation_failed" }` if
    /// the job itself failed.
    pub async fn generate(
        &self,
        request: &GenerateRequest,
        on_progress: Option<&dyn Fn(&Job)>,
    ) -> Result<Job> {
        let queued = self.submit(request).await?;
        let job = self.wait_for_job(&queued.id, on_progress).await?;
        if job.state == "failed" {
            return Err(Error::Api {
                code: "generation_failed".into(),
                message: job.error.unwrap_or_else(|| "unknown error".into()),
                status: 500,
            });
        }
        Ok(job)
    }

    // -- images -------------------------------------------------------------

    pub async fn history(&self) -> Result<Vec<HistoryImage>> {
        #[derive(Deserialize)]
        struct Wrapper {
            images: Vec<HistoryImage>,
        }
        let wrapper: Wrapper = self.get("/images").await?;
        Ok(wrapper.images)
    }

    /// Raw bytes of a generated image.
    pub async fn image_bytes(&self, filename: &str) -> Result<Vec<u8>> {
        let response = self
            .http
            .get(self.url(&format!("/images/{filename}")))
            .header("X-API-Key", &self.api_key)
            .send()
            .await?;
        let status = response.status();
        let bytes = response.bytes().await?;
        if !status.is_success() {
            return Err(Error::Api {
                code: "not_found".into(),
                message: format!("could not fetch {filename}"),
                status: status.as_u16(),
            });
        }
        Ok(bytes.to_vec())
    }

    /// Download an image to `dir`, returning the local path.
    pub async fn download(&self, filename: &str, dir: impl AsRef<Path>) -> Result<PathBuf> {
        let bytes = self.image_bytes(filename).await?;
        let dir = dir.as_ref();
        tokio::fs::create_dir_all(dir).await?;
        let base = Path::new(filename)
            .file_name()
            .unwrap_or_else(|| filename.as_ref());
        let path = dir.join(base);
        tokio::fs::write(&path, bytes).await?;
        Ok(path)
    }

    /// Copy an image into `.saved/`, beyond archive and delete-today.
    pub async fn save_image(&self, filename: &str) -> Result<serde_json::Value> {
        self.post_empty(&format!("/images/{filename}/save")).await
    }

    pub async fn delete_image(&self, filename: &str) -> Result<serde_json::Value> {
        self.delete(&format!("/images/{filename}")).await
    }

    /// Permanently delete all of today's output.
    pub async fn delete_today(&self) -> Result<serde_json::Value> {
        self.delete("/images").await
    }

    pub async fn archive(&self) -> Result<serde_json::Value> {
        self.post_empty("/archive").await
    }

    // -- reference images ---------------------------------------------------

    pub async fn import_url(&self, url: &str) -> Result<ImportedImage> {
        self.post_json("/imports/url", &serde_json::json!({ "url": url }))
            .await
    }

    pub async fn import_path(&self, path: &str) -> Result<ImportedImage> {
        self.post_json("/imports/path", &serde_json::json!({ "path": path }))
            .await
    }

    /// Convert a camera RAW file (NEF/DNG/CR3/...) the browser cannot decode.
    pub async fn import_raw(&self, path: impl AsRef<Path>) -> Result<ImportedImage> {
        let path = path.as_ref();
        let bytes = tokio::fs::read(path).await?;
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("upload.raw")
            .to_string();
        let part = reqwest::multipart::Part::bytes(bytes).file_name(name);
        let form = reqwest::multipart::Form::new().part("file", part);
        self.send(self.http.post(self.url("/imports/raw")).multipart(form))
            .await
    }

    pub async fn browse(&self, dir: Option<&str>) -> Result<DirListing> {
        let path = match dir {
            Some(d) => format!("/files?dir={}", urlencode(d)),
            None => "/files".to_string(),
        };
        self.get(&path).await
    }

    // -- vision-model jobs --------------------------------------------------

    /// Start a `describe`, `boost`, or `critique` job; returns the id to poll.
    pub async fn vlm_start(&self, task: &str, fields: serde_json::Value) -> Result<String> {
        let mut body = fields;
        body["task"] = serde_json::Value::String(task.into());
        let started: VlmStarted = self.post_json("/vlm/jobs", &body).await?;
        Ok(started.id)
    }

    /// Poll a vision job. A model-side failure surfaces as
    /// `Error::Api { code: "vlm_failed", status: 502 }`.
    pub async fn await_vlm(&self, id: &str) -> Result<serde_json::Value> {
        let deadline = Instant::now() + Duration::from_secs(900);
        while Instant::now() < deadline {
            let poll: VlmPoll = self.get(&format!("/vlm/jobs/{id}")).await?;
            if poll.done {
                return Ok(poll.result.unwrap_or(serde_json::Value::Null));
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        Err(Error::Timeout(format!("vlm job {id}")))
    }

    /// Photo(s) in, a prompt that would recreate them out.
    pub async fn describe(&self, data_urls: &[String], think: bool) -> Result<String> {
        let id = self
            .vlm_start(
                "describe",
                serde_json::json!({ "images": data_urls, "think": think }),
            )
            .await?;
        let result = self.await_vlm(&id).await?;
        Ok(result["prompt"].as_str().unwrap_or_default().to_string())
    }

    /// Rewrite a draft prompt into the loaded model's idiom. `level` 1-5 sets
    /// how far the rewrite may depart from the draft.
    pub async fn boost(&self, prompt: &str, level: u32, has_image: bool) -> Result<String> {
        let id = self
            .vlm_start(
                "boost",
                serde_json::json!({
                    "prompt": prompt,
                    "level": level,
                    "has_image": has_image,
                }),
            )
            .await?;
        let result = self.await_vlm(&id).await?;
        Ok(result["prompt"].as_str().unwrap_or_default().to_string())
    }

    /// Compare an edit's output against its reference and propose a revision.
    pub async fn critique(
        &self,
        direction: &str,
        ref_image: &str,
        output_filename: &str,
    ) -> Result<serde_json::Value> {
        let id = self
            .vlm_start(
                "critique",
                serde_json::json!({
                    "direction": direction,
                    "prompt": direction,
                    "ref_image": ref_image,
                    "output_filename": output_filename,
                }),
            )
            .await?;
        self.await_vlm(&id).await
    }

    // -- multi-model runs ---------------------------------------------------

    /// One prompt across several configs in turn, sharing a seed. Each config
    /// change restarts the server, so the run is file-backed and survives.
    pub async fn multi_run(
        &self,
        prompt: &str,
        configs: &[u32],
        steps: Option<u32>,
    ) -> Result<MultiRunStarted> {
        let mut body = serde_json::json!({ "prompt": prompt, "configs": configs });
        if let Some(steps) = steps {
            body["steps"] = serde_json::json!(steps);
        }
        self.post_json("/multi-runs", &body).await
    }

    /// The active run, or `None` when there isn't one.
    pub async fn multi_run_status(&self) -> Result<Option<MultiRunState>> {
        match self.get::<MultiRunState>("/multi-runs/current").await {
            Ok(state) => Ok(Some(state)),
            Err(e) if e.code() == Some("not_found") => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub async fn cancel_multi_run(&self) -> Result<()> {
        let _: serde_json::Value = self.delete("/multi-runs/current").await?;
        Ok(())
    }
}

/// Percent-encode a query value. Kept local so the crate needs no extra
/// dependency for the one place it matters.
fn urlencode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}
