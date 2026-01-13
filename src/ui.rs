use std::rc::Rc;

use yew::prelude::*;
use yew::{html, Component, Context, Html};
use web_sys::HtmlSelectElement;
use gloo_timers::future::sleep;
use std::time::Duration;

use crate::ai::AI;
use crate::logic::{Board, Direction, Pawn, Position, Color};

const SCALING: u32 = 80;

const MARGIN: u32 = 5;
pub enum Msg {
    PawnClick(usize),
    DirectionClick(Direction),
    Restart,
    AiGreen,
    AiYellow,
    AiMoveReady(Option<(usize, Direction)>),
    SetDifficulty(usize),
}

pub struct App {
    board: Board,
    state: Rc<AppState>,
    ai: Option<Color>,
    ai_depth: usize,
    ai_thinking: bool,
    selected_pawn: Option<usize>,
}

#[derive(Clone, PartialEq)]
pub struct AppState {
    pawn_clicked: Callback<usize>,
    direction_clicked: Callback<Direction>,
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        let pawn_clicked = ctx.link().callback(Msg::PawnClick);
        let direction_clicked = ctx.link().callback(Msg::DirectionClick);
        let state = Rc::new(AppState {
            pawn_clicked,
            direction_clicked,
        });

        let board = Board::default_new();
        Self {
            board,
            state,
            ai: None,
            ai_depth: 5,
            ai_thinking: false,
            selected_pawn: None,
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::SetDifficulty(depth) => {
                self.ai_depth = depth;
            }
            Msg::PawnClick(pawn_index) => {
                if self.board.next_player == Some(self.board.pawns[pawn_index].color.clone()){
                    self.selected_pawn = Some(pawn_index);
                }
            }
            Msg::DirectionClick(direction) => {
                if let Some(pawn_index) = self.selected_pawn {
                    let mut new_board = self.board.clone();
                    if new_board.move_pawn_until_blocked(pawn_index, &direction) {
                        self.board = new_board;
                        self.selected_pawn = None;

                        if let Some(ai_color) = &self.ai {
                            // Set AI thinking state
                            self.ai_thinking = true;
                            
                            // Clone data needed for the async task
                            let board = self.board.clone();
                            let ai_color = ai_color.clone();
                            let ai_depth = self.ai_depth;
                            
                            // Spawn async task to calculate AI move
                            let link = ctx.link().clone();
                            wasm_bindgen_futures::spawn_local(async move {
                                // Small delay to allow browser to render player's move first
                                sleep(Duration::from_millis(50)).await;
                                let ai_move = AI::new(ai_color, board, ai_depth).ai_play();
                                link.send_message(Msg::AiMoveReady(ai_move));
                            });
                            
                            return true;
                        }
                    }
                }
            }
            Msg::AiMoveReady(ai_move) => {
                self.ai_thinking = false;
                if let Some((ai_pawn_index, ai_direction)) = ai_move {
                    self.board.move_pawn_until_blocked(ai_pawn_index, &ai_direction);
                }
            }
            Msg::Restart => {
                self.board = Board::default_new();
                self.selected_pawn = None;
                self.ai = None;
                self.ai_thinking = false;
            }
            Msg::AiGreen => {
                self.ai = Some(Color::Yellow);
                self.selected_pawn = None;
                
                // Set AI thinking state
                self.ai_thinking = true;
                
                // Clone data needed for the async task
                let board = self.board.clone();
                let ai_depth = self.ai_depth;
                
                // Spawn async task to calculate AI move
                let link = ctx.link().clone();
                wasm_bindgen_futures::spawn_local(async move {
                    // Small delay to allow browser to render board state first
                    sleep(Duration::from_millis(50)).await;
                    let ai_move = AI::new(Color::Yellow, board, ai_depth).ai_play();
                    link.send_message(Msg::AiMoveReady(ai_move));
                });
                
                return true;
            }
            Msg::AiYellow => {
                self.ai = Some(Color::Green);
                self.selected_pawn = None;
                
                // Set AI thinking state
                self.ai_thinking = true;
                
                // Clone data needed for the async task
                let board = self.board.clone();
                let ai_depth = self.ai_depth;
                
                // Spawn async task to calculate AI move
                let link = ctx.link().clone();
                wasm_bindgen_futures::spawn_local(async move {
                    // Small delay to allow browser to render board state first
                    sleep(Duration::from_millis(50)).await;
                    let ai_move = AI::new(Color::Green, board, ai_depth).ai_play();
                    link.send_message(Msg::AiMoveReady(ai_move));
                });
                
                return true;
            }
        }
        true
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let app_state = self.state.clone();
        let next_player_text = match self.board.winner() {
            Some(Color::Green) => "Green wins!".to_string(),
            Some(Color::Yellow) => "Yellow wins!".to_string(),
            None => {
                if self.ai_thinking {
                    "🤔 AI is thinking...".to_string()
                } else {
                    match self.board.next_player {
                        Some(Color::Green) => "Green's turn".to_string(),
                        Some(Color::Yellow) => "Yellow's turn".to_string(),
                        None => String::new(),
                    }
                }
            }
        };
        
        // Configuration controls (AI selection, restart)
        let config_view = html! {
            <div class="config-controls">
                <button onclick={ctx.link().callback(|_| Msg::Restart)}>{ "Restart Game" }</button>
                <div class="difficulty-selector">
                    <label>{ "AI Difficulty: " }</label>
                    <select
                        onchange={ctx.link().callback(|e: Event| {
                            let input: HtmlSelectElement = e.target_unchecked_into();
                            Msg::SetDifficulty(input.value().parse().unwrap_or(3))
                        })}
                    >
                    <option value="1" selected={self.ai_depth == 1}>{ "Very Easy (Depth: 1)" }</option>
                    <option value="2" selected={self.ai_depth == 2}>{ "Easy (Depth: 2)" }</option>
                    <option value="3" selected={self.ai_depth == 3}>{ "Medium (Depth: 3)" }</option>
                    <option value="4" selected={self.ai_depth == 4}>{ "Hard (Depth: 4)" }</option>
                    <option value="5" selected={self.ai_depth == 5}>{ "Very Hard (Depth: 5)" }</option>
                    <option value="6" selected={self.ai_depth == 6}>{ "Expert (Depth: 6)" }</option>
                    <option value="7" selected={self.ai_depth == 7}>{ "Master (Depth: 7)" }</option>
                    </select>
                </div>
                <button onclick={ctx.link().callback(|_| Msg::AiGreen)}>{ "Play against AI as Green" }</button>
                <button onclick={ctx.link().callback(|_| Msg::AiYellow)}>{ "Play against AI as Yellow" }</button>
            </div>
        };

        // Game board and pawns
        let game_view = html! {
            <div class="game-container">
                <BoardView board={self.board.clone()} selected_pawn={self.selected_pawn} />

                // Direction buttons positioned around selected pawn
                {self.render_direction_buttons(ctx)}
            </div>
        };

        html! {
            <ContextProvider<Rc<AppState>> context={app_state}>
                <div class="app-container">
                    {config_view}
                    <h2>{ next_player_text }</h2>
                    {game_view}
                </div>
            </ContextProvider<Rc<AppState>>>
        }
    }
}

impl App {
    fn render_direction_buttons(&self, ctx: &Context<Self>) -> Html {
        if let Some(pawn_index) = self.selected_pawn {
            let pawn = &self.board.pawns[pawn_index];
            let valid_directions = self.board.get_valid_directions(pawn_index);

            let scaling_i32 = SCALING as i32;
            let controls_size = 180;

            let top = 50 + i32::from(pawn.position.row) * scaling_i32 - (controls_size - scaling_i32) / 2;
            let left = ((f64::from(pawn.position.column) + 0.5 - f64::from(self.board.number_of_columns) / 2.0 ) * f64::from(SCALING)) as i32;

            html! {
                <div class="direction-controls" style={format!("position: relative; top: {}px; left: {}px;", top, left)}>
                    <div class="dir-row">
                        {self.render_direction_button(ctx, Direction::UpLeft, &valid_directions, "↖")}
                        {self.render_direction_button(ctx, Direction::Up, &valid_directions, "↑")}
                        {self.render_direction_button(ctx, Direction::UpRight, &valid_directions, "↗")}
                    </div>
                    <div class="dir-row">
                        {self.render_direction_button(ctx, Direction::Left, &valid_directions, "←")}
                        <div class="dir-spacer"></div>
                        {self.render_direction_button(ctx, Direction::Right, &valid_directions, "→")}
                    </div>
                    <div class="dir-row">
                        {self.render_direction_button(ctx, Direction::DownLeft, &valid_directions, "↙")}
                        {self.render_direction_button(ctx, Direction::Down, &valid_directions, "↓")}
                        {self.render_direction_button(ctx, Direction::DownRight, &valid_directions, "↘")}
                    </div>
                </div>
                }
        } else {
            html! {}
        }
    }

    fn render_direction_button(&self, ctx: &Context<Self>, direction: Direction, valid_directions: &Vec<Direction>, symbol: &str) -> Html {
        let is_valid = valid_directions.contains(&direction);
        if is_valid {
            html! {
                <button
                    class="dir-btn"
                    onclick={ctx.link().callback(move |_| Msg::DirectionClick(direction.clone()))}
                >
                    {symbol}
                </button>
            }
        } else {
            html! {
                <div class="dir-spacer"></div>
            }
        }
    }
}

pub struct PawnView{
    state: Rc<AppState>,
    _listener: ContextHandle<Rc<AppState>>
}

#[derive(Clone, Properties, PartialEq)]
pub struct PawnComponent {
    pub pawn: Pawn,
    pub position: Position,
    pub index: usize,
    pub selected: bool,
}

pub enum PawnMsg {
    ContextChanged(Rc<AppState>),
}

impl Component for PawnView {
    type Message = PawnMsg;
    type Properties = PawnComponent;
 
    fn create(ctx: &Context<Self>) -> Self {
        let (state, _listener) = ctx
            .link()
            .context::<Rc<AppState>>(ctx.link().callback(PawnMsg::ContextChanged))
            .expect("context to be set");

        Self { state, _listener }
    }

    fn update(&mut self, _ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            PawnMsg::ContextChanged(state) => {
                self.state = state;
                true
            }
        }
    }
 
    fn view(&self, ctx: &Context<Self>) -> Html {
        let my_index = ctx.props().index;
        let onclick = self.state.pawn_clicked.reform(move |_| my_index);

        // Add selection highlight
        let border_style = if ctx.props().selected {
            "border: 3px solid red;"
        } else {
            "border: 2px solid black;"
        };

        html! {
            <div
                onclick={onclick}
                style={format!(
                    "width: {}px; height: {}px; background-color: {}; position: absolute; top: {}px; left: {}px; border-radius: 50%; {}; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: bold; color: black;",
                    SCALING - MARGIN * 2,
                    SCALING - MARGIN * 2,
                    match ctx.props().pawn.color {
                        Color::Green => "green",
                        Color::Yellow => "yellow",
                    },
                    u32::from(ctx.props().position.row) * SCALING + MARGIN - 2,
                    u32::from(ctx.props().position.column) * SCALING + MARGIN - 2,
                    border_style,
                )}
            >
                //{my_index}
            </div>
        }
    }
}

pub struct BoardView;

#[derive(Clone, Properties, PartialEq)]
pub struct BoardComponent {
    pub board: Board,
    pub selected_pawn: Option<usize>,
}
 
impl Component for BoardView {
    type Message = ();
    type Properties = BoardComponent;

    fn create(_ctx: &Context<Self>) -> Self {
        Self {}
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let mut pawns = Vec::new();
        for (index, pawn) in ctx.props().board.pawns.iter().enumerate() {
            pawns.push(html! {
                <PawnView
                    pawn={pawn.clone()}
                    position={pawn.position.clone()}
                    index={index}
                    selected={ctx.props().selected_pawn == Some(index)}
                />
            });
        }
        html! {
            <div style={format!(
                "position: absolute; top: {}px; width: {}px; height: {}px; background-image: linear-gradient(0deg, #e0e0e0 1px, transparent 1px), linear-gradient(90deg, #e0e0e0 1px, transparent 1px); background-size: {}px {}px; background-position: 0 0; border: 1px solid #e0e0e0;",
                50,
                SCALING * ctx.props().board.number_of_columns as u32,
                SCALING * ctx.props().board.number_of_rows as u32,
                SCALING, SCALING
            )}>
                {pawns}
            </div>
        }
    }
}
