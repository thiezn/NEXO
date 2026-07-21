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
		'queued',
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
	unloading_model_id TEXT,
	error_message TEXT,
	created_at TEXT NOT NULL,
	preparing_started_at TEXT,
	node_selected_at TEXT,
	model_loading_started_at TEXT,
	in_progress_at TEXT,
	completed_at TEXT,
	failed_at TEXT,
	last_state_changed_at TEXT NOT NULL,
	CHECK ((node_client_id IS NULL) = (node_device_id IS NULL)),
	CHECK (
		(run_state IN ('queued', 'preparing_context') AND node_client_id IS NULL AND model_id IS NULL AND unloading_model_id IS NULL)
		OR (run_state = 'unloading_model' AND node_client_id IS NOT NULL AND model_id IS NOT NULL AND unloading_model_id IS NOT NULL)
		OR (run_state IN ('loading_model', 'in_progress', 'completed') AND node_client_id IS NOT NULL AND model_id IS NOT NULL AND unloading_model_id IS NULL)
		OR run_state = 'failed'
	),
	CHECK ((run_state = 'failed') = (error_message IS NOT NULL)),
	FOREIGN KEY (node_client_id, node_device_id) REFERENCES nodes(client_id, device_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS agent_jobs (
	queue_position INTEGER PRIMARY KEY AUTOINCREMENT,
	operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id) ON DELETE CASCADE,
	job_kind TEXT NOT NULL CHECK (job_kind IN ('run_inference')),
	scheduler_state TEXT NOT NULL CHECK (scheduler_state IN ('runnable', 'waiting', 'completed', 'failed')),
	scheduled_for TEXT,
	waiting_since TEXT,
	wait_deadline TEXT,
	failure_message TEXT,
	created_at TEXT NOT NULL,
	updated_at TEXT NOT NULL,
	finished_at TEXT,
	CHECK (
		(scheduler_state = 'runnable' AND waiting_since IS NULL AND wait_deadline IS NULL AND finished_at IS NULL)
		OR (scheduler_state = 'waiting' AND waiting_since IS NOT NULL AND wait_deadline IS NOT NULL AND finished_at IS NULL)
		OR (scheduler_state = 'completed' AND failure_message IS NULL AND finished_at IS NOT NULL)
		OR (scheduler_state = 'failed' AND failure_message IS NOT NULL AND finished_at IS NOT NULL)
	)
);

CREATE TABLE IF NOT EXISTS node_job_leases (
	node_client_id TEXT NOT NULL,
	node_device_id TEXT NOT NULL,
	operation_id TEXT NOT NULL UNIQUE REFERENCES agent_jobs(operation_id) ON DELETE CASCADE,
	acquired_at TEXT NOT NULL,
	disconnected_at TEXT,
	disconnect_expires_at TEXT,
	CHECK ((disconnected_at IS NULL) = (disconnect_expires_at IS NULL)),
	PRIMARY KEY (node_client_id, node_device_id),
	FOREIGN KEY (node_client_id, node_device_id) REFERENCES nodes(client_id, device_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_agent_jobs_runnable
	ON agent_jobs(scheduler_state, scheduled_for, queue_position);

CREATE INDEX IF NOT EXISTS idx_agent_jobs_wait_deadline
	ON agent_jobs(scheduler_state, wait_deadline);

CREATE INDEX IF NOT EXISTS idx_node_job_leases_expiry
	ON node_job_leases(disconnect_expires_at);

CREATE INDEX IF NOT EXISTS idx_inference_runs_state
	ON inference_runs(run_state, last_state_changed_at);

CREATE INDEX IF NOT EXISTS idx_inference_intents_session
	ON inference_intents(session_id, created_at);
