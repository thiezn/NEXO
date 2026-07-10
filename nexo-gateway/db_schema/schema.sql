PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;
PRAGMA cache_size = -64000;

CREATE TABLE IF NOT EXISTS tool_definitions (
	name TEXT PRIMARY KEY,
	description TEXT NOT NULL,
	parameters_json TEXT NOT NULL CHECK (json_valid(parameters_json)),
	contract_version TEXT,
	execution_constraints_json TEXT NOT NULL CHECK (json_valid(execution_constraints_json)),
	created_at TEXT NOT NULL,
	updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
	client_id TEXT NOT NULL,
	device_id TEXT NOT NULL,
	connection_state TEXT NOT NULL CHECK (connection_state IN ('connected', 'disconnected')),
	first_connected_at TEXT NOT NULL,
	last_state_changed_at TEXT NOT NULL,
	last_disconnected_at TEXT,
	connected_at TEXT NOT NULL,
	PRIMARY KEY (client_id, device_id)
);

CREATE TABLE IF NOT EXISTS nodes (
	client_id TEXT NOT NULL,
	device_id TEXT NOT NULL,
	connection_state TEXT NOT NULL CHECK (connection_state IN ('connected', 'disconnected')),
	node_state TEXT NOT NULL CHECK (node_state IN (
		'idle',
		'loading_model',
		'unloading_model',
		'running_inference',
		'running_tool_call'
	)),
	first_connected_at TEXT NOT NULL,
	last_state_changed_at TEXT NOT NULL,
	last_disconnected_at TEXT,
	connected_at TEXT NOT NULL,
	PRIMARY KEY (client_id, device_id)
);

CREATE TABLE IF NOT EXISTS user_tools (
	user_client_id TEXT NOT NULL,
	user_device_id TEXT NOT NULL,
	tool_name TEXT NOT NULL REFERENCES tool_definitions(name) ON DELETE RESTRICT,
	PRIMARY KEY (user_client_id, user_device_id, tool_name),
	FOREIGN KEY (user_client_id, user_device_id) REFERENCES users(client_id, device_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS node_tools (
	node_client_id TEXT NOT NULL,
	node_device_id TEXT NOT NULL,
	tool_name TEXT NOT NULL REFERENCES tool_definitions(name) ON DELETE RESTRICT,
	PRIMARY KEY (node_client_id, node_device_id, tool_name),
	FOREIGN KEY (node_client_id, node_device_id) REFERENCES nodes(client_id, device_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS node_models_on_disk (
	node_client_id TEXT NOT NULL,
	node_device_id TEXT NOT NULL,
	model_id TEXT NOT NULL,
	PRIMARY KEY (node_client_id, node_device_id, model_id),
	FOREIGN KEY (node_client_id, node_device_id) REFERENCES nodes(client_id, device_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS node_models_in_memory (
	node_client_id TEXT NOT NULL,
	node_device_id TEXT NOT NULL,
	model_id TEXT NOT NULL,
	PRIMARY KEY (node_client_id, node_device_id, model_id),
	FOREIGN KEY (node_client_id, node_device_id) REFERENCES nodes(client_id, device_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS operations (
	operation_id TEXT PRIMARY KEY,
	user_client_id TEXT NOT NULL,
	user_device_id TEXT NOT NULL,
	created_at TEXT NOT NULL,
	UNIQUE (operation_id, user_client_id, user_device_id),
	FOREIGN KEY (user_client_id, user_device_id) REFERENCES users(client_id, device_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS inference_intents (
	operation_id TEXT PRIMARY KEY REFERENCES operations(operation_id) ON DELETE CASCADE,
	session_id TEXT NOT NULL,
	operation_kind TEXT NOT NULL CHECK (operation_kind IN (
		'multi_modal',
		'embed',
		'generate_image',
		'generate_speech',
		'tokenize',
		'detokenize'
	)),
	model_selection_json TEXT NOT NULL CHECK (json_valid(model_selection_json)),
	intent_json TEXT NOT NULL CHECK (json_valid(intent_json)),
	created_at TEXT NOT NULL,
	updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inference_runs (
	operation_id TEXT PRIMARY KEY REFERENCES operations(operation_id) ON DELETE CASCADE,
	run_state TEXT NOT NULL CHECK (run_state IN (
		'preparing_context',
		'unloading_model',
		'loading_model',
		'in_progress',
		'completed',
		'failed'
	)),
	node_client_id TEXT,
	node_device_id TEXT,
	model_id TEXT,
	error_message TEXT,
	created_at TEXT NOT NULL,
	preparing_started_at TEXT NOT NULL,
	node_selected_at TEXT,
	model_loading_started_at TEXT,
	in_progress_at TEXT,
	completed_at TEXT,
	failed_at TEXT,
	last_state_changed_at TEXT NOT NULL,
	FOREIGN KEY (node_client_id, node_device_id) REFERENCES nodes(client_id, device_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS agent_job_queue (
	queue_position INTEGER PRIMARY KEY AUTOINCREMENT,
	operation_id TEXT NOT NULL REFERENCES operations(operation_id) ON DELETE CASCADE,
	user_client_id TEXT NOT NULL,
	user_device_id TEXT NOT NULL,
	job_kind TEXT NOT NULL CHECK (job_kind IN ('run_inference')),
	status TEXT NOT NULL CHECK (status IN ('queued', 'claimed', 'completed', 'failed')),
	attempt_count INTEGER NOT NULL DEFAULT 0,
	failure_message TEXT,
	enqueued_at TEXT NOT NULL,
	claimed_at TEXT,
	finished_at TEXT,
	FOREIGN KEY (operation_id, user_client_id, user_device_id) REFERENCES operations(operation_id, user_client_id, user_device_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_agent_job_queue_dequeue
	ON agent_job_queue(status, queue_position);

CREATE INDEX IF NOT EXISTS idx_inference_runs_state
	ON inference_runs(run_state, last_state_changed_at);

CREATE INDEX IF NOT EXISTS idx_inference_intents_session
	ON inference_intents(session_id, created_at);
